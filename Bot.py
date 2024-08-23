import os
import copy
import platform
import torch
import json
from pprint import pprint, pformat
from datetime import datetime
import time
import logging
import pandas as pd
import csv
from prettytable import PrettyTable
from util import Configure, UTIL,color
from agent import Agent
from actions.Action import *
from host import HOST

# tensorboard --logdir runs --host localhost --port 8896

class BOT:
    """ Deep RL Bot """

    def __init__(self,
                 env_file=None,
                 agent: str = "PPO",
                 config=None,
                 save=True,
                 note='',
                 **kwargs):
        self.host_name = platform.platform()
        self.env_file = env_file
        self.agent_name = agent
        if config:
            self.agent = Agent(name=agent, config=config).agent
        else:
            self.agent = Agent(name=agent).agent
        self.base_agent=''
        self.save_model=save
        self.note=note
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')





    def make_env(self, env_file=None):
        target_list = []
        import json
        with open(env_file, 'r', encoding='utf-8') as f:  # *********
            self.environment_data = json.loads(f.read())
            train_ip_list = []
            for host in self.environment_data:
                ip = host["ip"]
                assert ip not in train_ip_list, f"{ip} aready exist in {env_file}"
                train_ip_list.append(ip)
                vul = host["vulnerability"][0]
                if vul not in Action.Vul_name_set:
                    logging.error(f"host vul {vul} is not exploitable"
                                    )  #TODO:整体逻辑需要改进
                    continue
                t = HOST(ip,env_data=host)
                target_list.append(t)
        
        return target_list

    def train(self):
        UTIL.line_break(symbol='+', length=60)
        logging.info("Starting training")
        UTIL.line_break(symbol='+', length=60)
        env_name=self.env_file.parent.name+'/'+self.env_file.stem
        self.title=f"{self.agent_name}-{self.current_time}-{env_name}"
        UTIL.Running_title=self.title

        env = self.make_env(self.env_file)

        config_to_log=copy.deepcopy(self.agent.config.__dict__)
        config_to_log["Algo"]=self.agent_name
        config_to_log["action_set"]=Action.vul_hub_path.name
        config_to_log["env_name"]=self.env_file.stem
        config_to_log["base_agent"]=self.base_agent

        
        config_df=pd.DataFrame.from_dict(config_to_log,orient='index')
        logging.info(config_df)
        pprint(config_df)

        self.agent.train(target_list=env)
        
        print(f"{color.color_str(self.current_time, c=color.GREEN)} training complete.")

  

    