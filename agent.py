
from RL_Model.config import  SAC_Config
import sys
import os
# curr_path = os.path.dirname(__file__)
# parent_path = os.path.dirname(curr_path)
# sys.path.append(curr_path)
# sys.path.append(parent_path)  # add current terminal path to sys.path


class Agent():
    def __init__(self, name, config=None):
        self.name = name
        self.config = config
        
        if self.name == "SAC_AE":
            from RL_Model.SAC_AE import SAC_agent
            if not self.config:
                self.config = SAC_Config()
            self.agent = SAC_agent(cfg=self.config)
        
        else:
            self.agent = None
            print("please imput agent name")
            exit()
        print(f"--Running {name} agent--")

   