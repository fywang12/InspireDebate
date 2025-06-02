import json
import requests
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError
import backoff
from tqdm import tqdm
import time

def match_topics_and_dialogues(topic_path: str, affirmative_path: str, negative_path: str, topic_indices: list = None) -> list:
    """
    Read topic list, affirmative and negative dialogue files, and match them one by one, returning a list where each element is (topic, affirmative_dialogue, negative_dialogue).
    The three must have the same length, otherwise an exception will be thrown.

    Args:
        topic_path (str): Path to the topic JSON file
        affirmative_path (str): Path to the affirmative dialogue JSON file
        negative_path (str): Path to the negative dialogue JSON file
        topic_indices (list, optional): List of topic indices to read (e.g., [401,402]), otherwise all topics by default

    Returns:
        list of tuples: [(topic, affirmative_dialogue, negative_dialogue), ...]
    """
    try:
        # Read topic
        with open(topic_path, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        # Read affirmative dialogue
        with open(affirmative_path, 'r', encoding='utf-8') as f:
            affirmative_dialogues = json.load(f)
        # Read negative dialogue
        with open(negative_path, 'r', encoding='utf-8') as f:
            negative_dialogues = json.load(f)

        # Select topic range
        if topic_indices is not None:
            start_idx = topic_indices[0]
            topics = [topics[i] for i in topic_indices]
            affirmative_dialogues = [affirmative_dialogues[i-start_idx] for i in topic_indices]
            negative_dialogues = [negative_dialogues[i-start_idx] for i in topic_indices]

        # Check length consistency
        if not (len(topics) == len(affirmative_dialogues) == len(negative_dialogues)):
            raise ValueError(f"the length of topics, affirmative_dialogues and negative_dialogues are not equal: topics={len(topics)}, affirmative={len(affirmative_dialogues)}, negative={len(negative_dialogues)}")

        # Match
        matched = []
        for topic, aff, neg in zip(topics, affirmative_dialogues, negative_dialogues):
            matched.append((topic, aff, neg))
        return matched
    except Exception as e:
        print(f"An error occurred while matching topics and dialogues: {e}")
        return []

class InspireScore:
    def __init__(self, openai_api_key: str, serper_api_key: str = None, model: str = None, base_url: str = "https://ark.cn-beijing.volces.com/api/v3"):
        """Initialize the InspireScore evaluator.
        
        Args:
            openai_api_key (str): OpenAI API key
            serper_api_key (str, optional): Serper API key for web search. Defaults to None.
            base_url (str, optional): Base URL for the API. Defaults to "https://ark.cn-beijing.volces.com/api/v3".
        """
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=base_url
        )
        self.serper_api_key = serper_api_key
        self.model = model

    def load_debate_content(self, file_path: str) -> list:
        """Load debate content from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file containing debate content
            
        Returns:
            list: List of assistant arguments from the debate
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                dialogues = json.load(file)  # data is a list of dialogues for one topic
            all_topics_assistant_arguments = []  # all_topics_assistant_arguments is a list of assistant arguments for all dialogues
            for dialogue in dialogues:  # each dialogue is a list of messages
                assistant_arguments = []
                for entry in dialogue:
                    if entry['role'] == 'assistant':
                        assistant_arguments.append(entry['content'])
                all_topics_assistant_arguments.append(assistant_arguments)
            return all_topics_assistant_arguments
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            return []

    def evaluate_subjective(self, topic: str, debate_text: str) -> str:
        """Evaluate the debate from a subjective perspective.
        
        Args:
            topic (str): The debate topic
            debate_text (str): The debate content
            
        Returns:
            str: Evaluation results in JSON format
        """
        system_prompt = """
        You are an experienced debate judge tasked with evaluating debates. For each debate, you will assess both sides based on four key criteria: Emotional Appeal, Clarity of Argument and Reasoning, Logical Arrangement of Arguments, and Relevance to Debate Topic.

        For each of the four subdimensions, provide a score from 0 to 1 (with 0 being the lowest and 1 being the highest) for both the **Pro (Affirmative)** side and the **Con (Negative)** side. Additionally, provide a brief analysis for both sides for each subdimension.
        
        Scoring Criteria:
            1. **Emotional Appeal**  
                - How effectively does each side connect with the audience emotionally? Does the argument evoke empathy, passion, or values?
                - **0**: No emotional appeal. The argument feels cold or disconnected.
                - **1**: Highly engaging emotionally, strongly connects with the audience.

            2. **Clarity of Argument and Reasoning**  
                - Are the arguments clearly presented? Is the reasoning sound and easy to follow?
                - **0**: The arguments are unclear or confusing.
                - **1**: The arguments are well-structured and easy to understand.

            3. **Logical Arrangement of Arguments**  
                - Is the argument presented in a logical, coherent manner? Does each point flow into the next without confusion?
                - **0**: The arguments are disorganized and difficult to follow.
                - **1**: The arguments follow a clear and logical progression.

            4. **Relevance to Debate Topic**  
                - Does each argument directly address the debate topic? Are there any irrelevant points or off-topic distractions?
                - **0**: Arguments that stray far from the topic.
                - **1**: Every argument is focused and relevant to the topic.

        Please output the result in the following format:

        1. **Pro (Affirmative Side) Score**:
            - Emotional Appeal: [score]
            - Argument Clarity: [score]
            - Argument Arrangement: [score]
            - Relevance to Debate Topic: [score]
            - **Total Score**: [total score]

        2. **Con (Negative Side) Score**:
            - Emotional Appeal: [score]
            - Argument Clarity: [score]
            - Argument Arrangement: [score]
            - Relevance to Debate Topic: [score]
            - **Total Score**: [total score]

        3. **Winner**: [Pro/Con]
        4. **Reason**: [Provide detailed analysis based on the scores]
        """
        
        user_prompt = f"""
        Evaluate the debate on the topic: '{topic}'
        Debate analysis process and arguments are as follows:
        {debate_text}
        Provide a JSON formatted response with scores and comments for each criterion for both debaters.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=12288,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def evaluate_logical_validity(self, topic: str, debate_text: str) -> str:
        """Evaluate the logical validity of the debate.
        
        Args:
            topic (str): The debate topic
            debate_text (str): The debate content
            
        Returns:
            str: Evaluation results in JSON format
        """
        system_prompt = """
        Task: Logical Inference

        Input:
        <Reasoning and Analysis Process>: Provide a step-by-step analysis leading to the formulation of the argument.
        <Argument>: Summarize the primary argument derived from the analysis.

        Output:
            1. Convert Reasoning and Argument to First-Order Logic(FOL): Transform the reasoning and statements into formalized logic expressions using the following rules:
                1. Logical conjunction of expr1 and expr2: expr1 ∧ expr2
                2. Logical disjunction of expr1 and expr2: expr1 ∨ expr2
                3. Logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
                4. Logical negation of expr1: ¬expr1
                5. expr1 implies expr2: expr1 → expr2
                6. expr1 if and only if expr2: expr1 ↔ expr2
                7. Logical universal quantification: ∀x
                8. Logical existential quantification: ∃x
            2. Generate Inference Plan: Outline a plan to evaluate whether the conclusions logically follow from the premises using first-order logic inference rules.
            3. Solve Logic Puzzle: Determine the truth value (true, false, unknown) of each conclusion based on the premises and logical inferences.
                Make sure you carefully and fully understand the below requirements before execution the conclusion:
                1.Please clearly indicate whether the conclusion statement is true, false or unknown using curly bracket {true/false/unknown}!!! The answer will only be either true, false or unknown.
                The definition of the three options are:
                    True: A statement is "true" if it necessarily follows from the given premises using logical rules.
                    False: A statement is "false" if it is contradicted by the premises or its negation is logically inferred from them.
                    Unknown: A statement is "unknown" if there is insufficient information in the premises to determine its truth value conclusively.
                2. Make sure you must only use the premises to infer the conclusion. Do not use any information that is not exist or cannot be inferred from the premises.If some premise is semantically equal, such as "love the most" and "favorite", you can consider this as a valid assumption. You can make assumption to entity if it is very obvious but not logical relationship. For instance, an entity with an obvious human name can be inferred as a human.
                3. Make sure you abide the 16 provided first-order logic rules and formula when making logical inference. You need to clearly indicate what logic rules and formula you used.
                4. Please note that in first-order logic if there exists a conditional statement in the conclusion such as "If...", the if part will be considered as a premise. And if there is premise contradicts the if statement, you need to use the premise in the if statement as priority and neglect the contradicted one.
                5. Be careful with the parentheses. Make sure you following the rules such as Order of Operations (The order is usually: negation (¬), conjunction (and, ∧), disjunction (or, ∨), implication (→), and biconditional (↔). ), Nested Parentheses (The expression inside the innermost set of parentheses is evaluated first, then the next outer set, and so on.). 
                6. Make sure you not only access the premises in first-order logic, but also access its corresponding natural language format. The natural language format premises should be prioritized when there is inconsistent between natural language and first-order logic.
                7. When inferring new knowledge, please clear indicate which premises you used or the steps you refer to. For instance, if you use Premise 1 and a knowledge from Step 5, you should clearly indicate that "Combine Premise 1 and Step 5".
                8. You should also use natural language to explain the logical process in each step. Please also indicate the premises and steps you refer to when making the logical process.
        -----
        Example:

        Input:

            Reasoning and Analysis Process:
                1. Ethical Concerns: Human cloning raises profound ethical concerns about the sanctity of human life and the potential for exploitation. Without strict regulation, cloning could lead to commodifying human beings for purposes such as organ harvesting, which undermines the inherent value and dignity of life.
                2. Psychological Impact: Cloned individuals could face identity issues and social stigma due to being seen as copies of others, potentially causing psychological distress.
                3. Societal Implications: Cloning might disrupt cultural norms and legal systems by introducing a new class of individuals with ambiguous rights, furthering societal inequalities.
                4. Medical Risks: Cloning is prone to errors, with high risks of genetic abnormalities and birth defects, compromising the health of cloned individuals.
                5. Scientific Challenges: Cloning is an underdeveloped technology with unknown long-term effects, necessitating caution and further study before widespread adoption.
            Argument:
                We should ban human cloning because it poses significant ethical, psychological, societal, and medical challenges that outweigh any potential benefits. The potential for exploitation, identity issues, societal disruption, and health risks make cloning a morally and socially unacceptable practice. Moreover, the scientific uncertainties highlight the importance of regulation to prevent unintended consequences. A ban ensures human dignity is preserved and protects society from these risks.

        Output:

            1. Convert Reasoning and Argument to Logic expression:

            Predicates:
                    EthicalRisk(x) ::: x poses ethical risks.
                    PsychologicalRisk(x) ::: x poses psychological risks.
                    SocietalRisk(x) ::: x poses societal risks.
                    MedicalRisk(x) ::: x poses medical risks.
                    ScientificUncertainty(x) ::: x is scientifically uncertain.
                    ShouldBan(x) ::: x should be banned.
                    UndermineDignity(x) ::: x undermines the dignity of human life.
                    CommodifyHumans(x) ::: x commodifies human beings.

            Premises:
                1. HumanCloning → EthicalRisk(HumanCloning) ::: Human cloning poses ethical risks.
                2. EthicalRisk(HumanCloning) → (UndermineDignity(HumanCloning) ∧ CommodifyHumans(HumanCloning)) ::: Ethical risks include undermining dignity and commodifying humans.
                3. HumanCloning → PsychologicalRisk(HumanCloning) ::: Human cloning poses psychological risks.
                4. HumanCloning → SocietalRisk(HumanCloning) ::: Human cloning poses societal risks.
                5. HumanCloning → MedicalRisk(HumanCloning) ::: Human cloning poses medical risks.
                6. HumanCloning → ScientificUncertainty(HumanCloning) ::: Human cloning is scientifically uncertain.

            Conclusions:
                1. (EthicalRisk(HumanCloning) ∧ PsychologicalRisk(HumanCloning) ∧ SocietalRisk(HumanCloning) ∧ MedicalRisk(HumanCloning) ∧ ScientificUncertainty(HumanCloning)) → ShouldBan(HumanCloning) ::: Human cloning should be banned because it involves ethical, psychological, societal, medical, and scientific risks.
                2. (UndermineDignity(HumanCloning) ∨ CommodifyHumans(HumanCloning)) → ShouldBan(HumanCloning) ::: Cloning undermines dignity and commodifies humans, justifying a ban.
                3. (MedicalRisk(HumanCloning) ∧ ScientificUncertainty(HumanCloning)) → ShouldBan(HumanCloning) ::: The medical risks and scientific uncertainties of cloning necessitate a ban.

            2. Generate Inference Plan:

            Plan:
                1. Identify the Goal: Our goal is to assess the truth value of each conclusion statement (true, false, or unknown) based on the logical relationship and inference rules established by the given premises.
                2. Utilize Logical Inference Rules: Determine which logical inference rules (e.g., Modus Ponens, Conjunction, etc.) can be used to derive conclusions from the premises.
                3. Detailed Analysis of Each Conclusion:
                    Conclusion 1: Analyze if all individual risks (Ethical, Psychological, Societal, Medical, Scientific) that are stated to result from human cloning can collectively justify a ban on human cloning.
                    Use Conjunction and Modus Ponens to infer if the comprehensive risk leads to the necessity of a ban.
                    Conclusion 2: Evaluate if any of the specific risks (either UndermineDignity or CommodifyHumans) individually justifies a ban on human cloning.
                    Use Disjunction to infer if either condition being true would necessitate a ban.
                    Conclusion 3: Consider if the combination of Medical risks and Scientific uncertainties sufficiently argues for a ban.
                    Use Conjunction to deduce the combined effect of these two specific risks.
                4. Apply Premises to Conclusions:
                    For Conclusion 1:
                    Premises 1, 2, 3, 4, 5, and 6 provide a logical pathway through conjunction to infer the comprehensive risks. Check if this leads to the stated need for a ban through Modus Ponens.
                    For Conclusion 2:
                    Use Premise 2 directly, which links Ethical risks (that include Dignity and Commodification issues) to potentially banning human cloning.
                    For Conclusion 3:
                    Premises 5 and 6 establish both Medical and Scientific risks. Use Modus Ponens with the conjunction of these risks to infer the necessity of a ban.
                5. Evaluate Logical Connections:
                    Verify each step's logical validity based on the inference rules and premises. Ensure that there are no logical fallacies or gaps in the argumentative chain.
                6. Combine the Steps:
                    Integrate the information from all steps to form a reasoned argument for each conclusion.
                7. Conclude:
                    Determine whether each conclusion is true, false, or unknown based on the outcome of the logical deductions. Each conclusion should be evaluated independently to determine its validity.
                8. Final Step:
                    Document the outcomes and provide explanations for each conclusion's status (true, false, unknown) as inferred from the premises and logical rules used.
            This plan should provide a clear path to test the conclusion statement against the premises using first-order logic inference rules.
            
            3. Evaluate Conclusions: 
                Clearly indicate whether the conclusion statement is true, false or unknown using curly bracket.
                Conclusion 1: {true}
                Conclusion 2: {true}
                Conclusion 3: {true}
            
            This systematic approach ensures that the evaluation of conclusions is based on a structured analysis using first-order logic, adhering to the standards of logical rigor and clarity.
        """
        
        user_prompt = f"""
        Topic: {topic}
        
        Debate Text:
        {debate_text}
        
        Output the response in JSON format as per the provided system instructions.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=12288,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def extract_atomic_facts(self, topic: str, debate_text: str) -> str:
        """Extract atomic facts from the debate text.
        
        Args:
            topic (str): The debate topic
            debate_text (str): The debate content
            
        Returns:
            str: Extracted facts in JSON format
        """
        system_prompt = """
        You are tasked with breaking down reasoning processes and arguments into atomic facts. Follow these instructions:
        1. An atomic fact is a single, standalone statement containing one idea or piece of information.
        2. Each atomic fact should capture a distinct piece of information and avoid overlaps.
        3. For the reasoning process, break down each statement into separate facts labeled sequentially as fact-1, fact-2, and so on.
        4. For the argument, break down each reason provided into atomic facts labeled sequentially, continuing from the last reasoning fact.
        5. Provide the output in JSON format as follows:
        {
          "reasoning_process": [
            "fact-1: <reasoning atomic fact>",
            "fact-2: <reasoning atomic fact>",
            ...
          ],
          "argument": [
            "fact-<n>: <argument atomic fact>",
            "fact-<n+1>: <argument atomic fact>",
            ...
          ]
        }
        Ensure the numbering of facts is sequential and consistent across the reasoning process and argument sections.
        """
        
        user_prompt = f"""
        Topic: {topic}
        
        Reasoning Process and Argument:
        {debate_text}
        
        Break the reasoning process and argument into atomic facts according to the instructions. Provide the response in JSON format.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=12288,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def web_search(self, query: str) -> dict:
        """Perform a web search using the Serper API.
        
        Args:
            query (str): The search query
            
        Returns:
            dict: Search results
        """
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
        
    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=20)
    def generate_search_queries(self, fact_json: str) -> str:
        """Generate search queries for fact verification.
        
        Args:
            fact_json (str): JSON string containing facts
            
        Returns:
            str: Generated search queries
        """
        system_prompt = """
        You are an expert fact-checking assistant. Your task is to analyze the provided JSON content and generate relevant queries that should be searched on the internet (e.g., using Google) to validate the facts.
        - Carefully examine the JSON content and identify the main topics or themes that emerge from the claims.
        - Propose precise and actionable search queries that can help verify the claims.
        - Your response should only include the search queries in a clear and concise list, and must not exceed 2000 characters.
        """
        
        user_prompt = f"""
        Analyze the following JSON content and generate search queries to validate the claims:
        {fact_json}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=12288,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def verify_facts(self, search_results: dict, fact_json: str) -> str:
        """Verify facts using web search results.
        
        Args:
            search_results (dict): Web search results
            fact_json (str): JSON string containing facts
            
        Returns:
            str: Verification results in JSON format
        """
        system_prompt = """
        You are an expert fact-checking assistant. Your task is to verify the provided facts in the JSON content by analyzing the given search results.

        - For each fact, determine if it is "true," "false," or "unknown" based on the evidence provided in the search results.
        - "true": Strong and reliable evidence supports the fact.
        - "false": Strong and reliable evidence disproves the fact.
        - "unknown": Evidence is insufficient or inconclusive to verify the fact.

        - Be specific and logical in your assessment, focusing on the factual accuracy of each claim.

        - If the search results are empty, rely on your existing knowledge to assess the factual accuracy of the claims.

        Output your analysis in the following JSON format:
        {
            "fact-1": "true/false/unknown",
            "fact-2": "true/false/unknown",
            ...
        }
        """
        
        user_prompt = f"""
        Given the following inputs:

        JSON Content: {fact_json}
        Search Results: {search_results}

        Analyze the search results (or, if empty, use your existing knowledge) to verify the facts in the JSON content. Provide your conclusions in the specified JSON format.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=12288,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def evaluate_debate(self, topic: str, affirmative_dialogue: list, negative_dialogue: list, output_file: str):
        """Evaluate a debate and save results to a JSON file.
        
        Args:
            topic (str): The debate topic
            affirmative_dialogue (list): The affirmative side's dialogue content
            negative_dialogue (list): The negative side's dialogue content
            output_file (str): Path to save the evaluation results
        """
        # Extract assistant's arguments
        affirmative_arguments = [entry['content'] for entry in affirmative_dialogue if entry['role'] == 'assistant']
        negative_arguments = [entry['content'] for entry in negative_dialogue if entry['role'] == 'assistant']
        
        # Combine arguments
        combined_arguments = []
        for aff, neg in zip(affirmative_arguments, negative_arguments):
            combined_arguments.append(aff)
            combined_arguments.append(neg)
        
        # Prepare debate text
        debate_text = ""
        for idx, argument in enumerate(combined_arguments):
            if idx % 2 == 0:
                debate_text += f"Affirmative Argument {idx//2 + 1}: {argument}\n"
            else:
                debate_text += f"Negative Argument {idx//2 + 1}: {argument}\n"
        
        # Perform evaluations
        subjective_eval = self.evaluate_subjective(topic, debate_text)
        
        # Evaluate each argument from both sides
        evaluation_results = {
            "topic": topic,
            "overall_subjective_evaluation": subjective_eval,
            "affirmative_evaluations": [],
            "negative_evaluations": []
        }
        
        # Evaluate affirmative arguments
        for idx, argument in enumerate(affirmative_arguments):
            # Subjective evaluation
            # subjective_eval = self.evaluate_subjective(topic, f"Affirmative Argument {idx + 1}: {argument}")
            
            # Logical validity evaluation
            logical_eval = self.evaluate_logical_validity(topic, argument)
            
            # Fact verification
            facts = self.extract_atomic_facts(topic, argument)
            queries = self.generate_search_queries(facts)
            search_results = self.web_search(queries)
            verification = self.verify_facts(search_results, facts)
            
            # Save evaluation results for this affirmative argument
            evaluation_results["affirmative_evaluations"].append({
                "argument_index": idx + 1,
                "argument": argument,
                # "subjective_evaluation": subjective_eval,
                "logical_evaluation": logical_eval,
                "fact_evaluation": {
                    "facts": facts,
                    "queries": queries,
                    "search_results": search_results,
                    "verification": verification
                }
            })
        
        # Evaluate negative arguments
        for idx, argument in enumerate(negative_arguments):
            # Subjective evaluation
            # subjective_eval = self.evaluate_subjective(topic, f"Negative Argument {idx + 1}: {argument}")
            
            # Logical validity evaluation
            logical_eval = self.evaluate_logical_validity(topic, argument)
            
            # Fact verification
            facts = self.extract_atomic_facts(topic, argument)
            queries = self.generate_search_queries(facts)
            search_results = self.web_search(queries)
            verification = self.verify_facts(search_results, facts)
            
            # Save evaluation results for this negative argument
            evaluation_results["negative_evaluations"].append({
                "argument_index": idx + 1,
                "argument": argument,
                # "subjective_evaluation": subjective_eval,
                "logical_evaluation": logical_eval,
                "fact_evaluation": {
                    "facts": facts,
                    "queries": queries,
                    "search_results": search_results,
                    "verification": verification
                }
            })
        
        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

def main():
    # Example usage
    evaluator = InspireScore(
        openai_api_key = "your openai api key",
        serper_api_key = "your serper api key",
        base_url="your base url",
        model="model name, e.g. gpt-4"
    )
    
    # Demo debate result
    affirmative_file_path = 'debate_dialogue_demo/20250119_debate_result_refute_vs_refute/Affirmative side_memory.json'
    negative_file_path = 'debate_dialogue_demo/20250119_debate_result_refute_vs_refute/Negative side_memory.json'
    topic_path = 'Debate_topic/all_topic_710.json'
    output_file = "inspirescore_results/debate_evaluation_results.json"
    
    # Match topics and dialogues
    matched_list = match_topics_and_dialogues(
        topic_path=topic_path,
        affirmative_path=affirmative_file_path,
        negative_path=negative_file_path,
        topic_indices=list(range(401, 402))  # Demo topic
    )
    
    # Evaluate each matched topic and dialogue
    for topic, aff_dialogue, neg_dialogue in matched_list:
        evaluator.evaluate_debate(
            topic=topic,
            affirmative_dialogue=aff_dialogue,
            negative_dialogue=neg_dialogue,
            output_file=output_file
        )

if __name__ == "__main__":
    main() 