import openai
from openai import OpenAI
import backoff
import time
import random
import requests
import json
from typing import Dict, List, Optional
# from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError #old api
from openai import RateLimitError, APIError, APIConnectionError
from .openai_utils import OutOfQuotaException, AccessTerminatedException
from .openai_utils import num_tokens_from_string, model2max_context
# set the supported models list
support_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314', 'ep-xxx','llama31-series'] #ep-xxx is Volcengine inference point, llama31-series is the llama3.1 series models

class Agent_debate:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0, url:str="http://0.0.0.0:8000/v1", serper_api_key: Optional[str] = None,openai_api_key: Optional[str] = None, debate_topic:str=None) -> None:
        self.model_name = model_name #model name of the debate player
        self.name = name #agent's debate position: affirmative, negative
        self.temperature = temperature
        self.memory_lst = [] #stores conversation history in the format required by the model (user messages, assistant responses, etc.)
        self.sleep_time = sleep_time
        self.client = None # openai client
        self.url = url
        self.serper_api_key = serper_api_key
        self.openai_api_key = openai_api_key
        self.debate_topic = debate_topic
        self.web_search_results = {}  # stores web search results

    # backoff decorator for automatic retry on specific exceptions
    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=20)
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        """make a query
        #  core method that sends a query to the OpenAI model using the stored conversation
        Args:
            messages (list[dict]): chat history in turbo format
            max_tokens (int): max token in api call
            api_key (str): openai api key
            temperature (float): sampling temperature

        Raises:
            OutOfQuotaException: the apikey has out of quota
            AccessTerminatedException: the apikey has been ban

        Returns:
            str: the return msg
        """
        #self.client = OpenAI(api_key=api_key)
        base_url = self.url
        if ".com" in base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key="local_model_key", base_url=base_url)
        time.sleep(self.sleep_time) #sleep time, to prevent frequent api calls
        assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"#assert, if the model is not supported, an error will be reported
        try:
            if self.model_name in support_models:
                # openai-gpt series
                # response = self.client.chat.completions.create(
                #     model=self.model_name,
                #     messages=messages,
                #     temperature=temperature,
                #     max_tokens=max_tokens
                # )
                # llama series can be executed without changing the parameters
                response = self.client.chat.completions.create(
                    model=self.model_name, #for some models, need to specify the model access point explicitly
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                gen = response.choices[0].message.content
            return gen

        except RateLimitError as e:
            if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
                raise OutOfQuotaException(api_key)
            elif "Your access was terminated due to violation of our policies" in e.user_message:
                raise AccessTerminatedException(api_key)
            else:
                raise e

    def set_meta_prompt(self, meta_prompt: str):
        """Set the meta_prompt

        Args:
            meta_prompt (str): the meta prompt
        """
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        """Add an new event in the memory

        Args:
            event (str): string that describe the event.
        """
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
        """
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")
    
    # save the conversation history to a JSON file
    def save_memory_to_json(self):
        """Save memory_lst to a JSON file named after the agent's name."""
        filename = f"debate_result/{self.name}_memory.json"  # modify the file name generation path
        with open(filename, "a", encoding="utf-8") as file:
            json.dump(self.memory_lst, file, ensure_ascii=False, indent=4)
            file.write('\n')  # add a newline
        print(f"Memory saved to {filename}")

    def ask(self, temperature: float=None):
        # Web-RAG processing for the debate process [optional]
        debate_topic = self.debate_topic  # debate topic
        position = self.name      # position
        keywords = self.generate_search_queries(debate_topic, position)# generate search keywords
        search_evidence = self.search_and_summary_concatenate(keywords)# get the summary of the search results and concatenate to the user prompt of memory_lst
        
        # calculate the token number and generate the answer
        num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
        max_token = model2max_context[self.model_name] - num_context_token
        
        return self.query(self.memory_lst, max_token, api_key=self.openai_api_key, temperature=temperature if temperature else self.temperature)

    def generate_search_queries(self, debate_topic: str, position: str) -> List[str]:
        # check if there is an opponent argument (the last user message in the memory)
        opponent_argument = None
        if len(self.memory_lst) > 2 and self.memory_lst[-1]["role"] == "user":
            opponent_argument = self.memory_lst[-1]["content"]

        # first stage: extract keywords
        system_prompt = """
        You are a professional debate assistant. Your task is to extract 1-3 precise search terms that will help gather factual evidence for the debate.

        Requirements:
        1. Generate exactly 1-3 keywords or phrases
        2. Each keyword should be specific and searchable
        3. Keywords should be concise (2-4 words each)
        4. Avoid overly broad or vague terms

        Output Format:
        Return a JSON array containing exactly 1-3 keywords, like this:
        ["keyword1", "keyword2", "keyword3"]
        """
        
        user_prompt = f"""
        Debate Topic: {debate_topic}
        Position: {position}
        {f"Opponent's Argument: {opponent_argument}" if opponent_argument else ""}

        Please generate 1-3 precise search keywords.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # generate search keywords
        keywords_response = self.query(messages, max_tokens=1000, api_key=self.openai_api_key, temperature=0.7)
        
        try:#if the return is json format, parse it, otherwise split by newline
            keywords = json.loads(keywords_response)
            keywords = keywords if isinstance(keywords, list) else [keywords] #if the return is not a list, convert it to a list
        except json.JSONDecodeError:
            keywords = [k.strip() for k in keywords_response.split('\n') if k.strip()]
        return keywords

    def web_search(self, query: str) -> Dict:
        if not self.serper_api_key:
            raise ValueError("Serper API key is required for web search")
            
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            'Content-Type': 'application/json'
        }
        data = {"q": query, "num": 3}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Web search error: {e}")
            return {}

    def search_and_summary_concatenate(self,keywords:List[str]) -> str:
        
        # run web search for each keyword, get 3 search results
        search_results = {}
        for keyword in keywords:
            results = self.web_search(keyword)
            if results:
                search_results[keyword] = results

        # summarize the search results
        summary = []
        for query, results in search_results.items():
            summary.append(f"Query: {query}")
            # add organic search results
            if 'organic' in results:
                for result in results['organic']:
                    summary.append(f"- {result.get('title', '')}")
                    summary.append(f"  {result.get('snippet', '')}")
            # add "People Also Ask" results[optional]
            if 'peopleAlsoAsk' in results:
                summary.append("\nRelated Questions:")
                for qa in results['peopleAlsoAsk']:
                    summary.append(f"- Q: {qa.get('question', '')}")
                    summary.append(f"  A: {qa.get('snippet', '')}")
            summary.append("\n")
            search_evidence = "\n".join(summary)

        # concatenate the search results to the last user prompt
        for i in range(len(self.memory_lst) - 1, -1, -1):
            if self.memory_lst[i]["role"] == "user":
                self.memory_lst[i]["content"] += f"\n\nRetrieved evidence:\n{search_evidence}"
                break

        return search_evidence

import re
import json
def extract_json(text, openai_key=None):
    """
    extract the JSON format data from the input text.
    use regex to extract the dictionary part, if failed, try to use eval method,
    if still failed, use GPT to generate the JSON format.
    """
    # use regex to extract the dictionary part
    match = re.search(r'{.*}', text)
    if match:#if the dictionary string is found
        dict_str = match.group()#get the dictionary string
        
        # try to use JSON to parse
        try:
            result_dict = json.loads(dict_str)
            return result_dict
        except json.JSONDecodeError:
            print("JSON decode failed, try to use eval to parse")
            
            # use eval to parse the json format
            try:
                result_dict = eval(dict_str)
                if isinstance(result_dict, dict):
                    return result_dict
                else:
                    print("eval failed, the result is not a dictionary")
                    return None
            except (SyntaxError, NameError):
                print("eval failed")
                return None
    else:
        print("the dictionary string is not found, use GPT and json.loads() to parse")
        
        # call the GPT model to generate the JSON format
        prompt = f"Extract JSON format content from the following text in this format: {{'Whether there is a winner': 'Yes or No', 'Supported Side': 'Affirmative or Negative', 'Reason': ''}}\n\nText:\n{text}"
        
        # set the api key and call the model
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     temperature=0,
        #     max_tokens=max_tokens
        # )
        
        # get the generated text content and try to parse it as a dictionary
        gpt_output = text#response.choices[0].text.strip()
        try:
            result_dict = json.loads(gpt_output)
            return result_dict
        except json.JSONDecodeError:
            print("GPT generated content is not a valid JSON format")
            return None
