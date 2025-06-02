# random.seed(0)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import random
import sys
import os
from codes.utils.agent import Agent_debate
from codes.utils.agent import extract_json

class DebatePlayer(Agent_debate): #inherit from agent, is the participant of the debate
    def __init__(self, model_name: str, name: str, temperature:float, openai_api_key: str,serper_api_key: str, sleep_time: float,url:None,debate_topic:str=None) -> None:
        #initialize the player
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time,url,debate_topic=debate_topic,openai_api_key=openai_api_key,serper_api_key=serper_api_key)

class Debate: #create and execute the whole debate process
    def __init__(self,
            model_name: str='gpt-3.5-turbo', #default：'gpt-3.5-turbo' 'llama31-8b-instruct'
            temperature: float=0, 
            num_players: int=2, 
            openai_api_key: str=None,
            serper_api_key: str=None,
            config: dict=None,
            max_round: int=3,
            sleep_time: float=0,
            debate_index: int=0
        ) -> None:

        self.model_name = model_name #model name
        self.temperature = temperature#temperature
        self.num_players = num_players#number of players
        self.openai_api_key = openai_api_key#api of model
        self.serper_api_key = serper_api_key#api of serper
        self.config = config#config file
        self.max_round = max_round # maximum rounds of debate
        self.sleep_time = sleep_time#sleep time
        self.debate_topic = config['debate_topic']#debate topic
        self.init_prompt() #initialize prompt
        # creat&init agents
        self.creat_agents() #create agents
        self.init_agents() #initialize agents

    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("affirmative_prompt")

    def creat_agents(self):
        # creates players
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, openai_api_key=self.openai_api_key,serper_api_key=self.serper_api_key, sleep_time=self.sleep_time,url=model_url,debate_topic=self.debate_topic) for name, model_url in zip(NAME_LIST, MODEL_URL_LIST)
        ]
        self.affirmative = self.players[0]  #affirmative player
        self.negative = self.players[1]  #negative player

    def init_agents(self): #when the agents are initialized, the first round of debate will be started
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])#meta_prompt is the system prompt
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])#meta_prompt is the system prompt
        
        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")
        self.affirmative.add_event(self.config['affirmative_prompt']) #add the user prompt to the memory of affirmative player
        self.aff_ans = self.affirmative.ask() #get the answer from the affirmative player
        self.affirmative.add_memory(self.aff_ans) #add the answer to the memory of affirmative player

        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))#add the user prompt to the memory of negative player
        self.neg_ans = self.negative.ask()#get the answer from the negative player
        self.negative.add_memory(self.neg_ans)#add the answer to the memory of negative player

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def run(self): #continue the debate
        for round in range(self.max_round - 1): #set the maximum number of rounds to 3
            print(f"===== Debate Round-{round+2} =====\n")
            self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
            self.aff_ans = self.affirmative.ask()
            self.affirmative.add_memory(self.aff_ans)

            self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
            self.neg_ans = self.negative.ask()
            self.negative.add_memory(self.neg_ans)
            self.config['success'] = True
            
        self.print_answer()
        # save memory
        for player in self.players:
            player.save_memory_to_json()

if __name__ == "__main__":
    openai_api_key = "Your-OpenAI-Api-Key"
    serper_api_key = "Your-Serper-Api-Key"
    model_name = "Your-Model-Name"
    # name list of the player, affirmative and negative
    NAME_LIST=[
        "Affirmative side",
        "Negative side"
    ]
    # model url list of the player
    MODEL_URL_LIST = [
        "https://ark.cn-beijing.volces.com/api/v3", #ark.cn-beijing.volces.com
        "https://ark.cn-beijing.volces.com/api/v3", #ark.cn-beijing.volces.com
        # "https://api.openai.com/v1", #gpt-4o...
        # "https://api.openai.com/v1", #gpt-4o...
        # "http://0.0.0.0:8000/v1",# local model...
        # "http://0.0.0.0:8001/v1",# local model...
    ]
    # get the absolute path of the current project
    current_script_path = os.path.abspath(__file__)
    _path = current_script_path.rsplit("/", 1)[0]
    # load the topics from the json file
    topics = json.load(open(f"{_path}/debate_topic/all_topic_710.json", "r"))
    # choose the topics to debate
    for topic in topics[200:201]:
        debate_topic = topic
        while debate_topic == "":
            debate_topic = input(f"\nEnter your debate topic: ")
        # some basic prompts for the debate were saved in the config4all.json
        config = json.load(open(f"{_path}/codes/utils/config4all.json", "r"))
        config['debate_topic'] = debate_topic
        # start the debate
        debate = Debate(model_name=model_name,num_players=2, max_round=2, openai_api_key=openai_api_key, serper_api_key=serper_api_key, config=config, temperature=0.7, sleep_time=0)
        debate.run()