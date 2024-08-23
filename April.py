import os
from datetime import datetime
import logging
from pathlib import Path
import torch
import random
import numpy as np

# tensorboard --logdir
# tensorboard --logdir runs --host localhost --port 8896
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ###指定此处为-1即可

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("--env_file",default="single\env-CVE-2018-11776.json",
                        help="training data set, e.g. train.json")
    parser.add_argument("--agent", default="SAC_AE", help="support PPO D3QN")
    parser.add_argument("--base", default="", type=str,help="support PPO D3QN")
    parser.add_argument("--save", type=bool, default=False, help="support PPO D3QN")
    parser.add_argument("--seed", type=int, default=0, help="support PPO D3QN")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    args = parser.parse_args()
    seed=args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    current_path=Path.cwd()

    from util import Configure, set_logger, UTIL
    from Bot import BOT 

    log_path = Configure.get('common', 'log_path')
    set_logger(os.path.join(log_path, 'April_train.log'))
    UTIL.show_banner()
    UTIL.line_break(length=80, symbol='=')
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logging.info(current_time)
    UTIL.line_break(length=80, symbol='=')

    scenario_path=current_path /"scenarios"
    args.env_file=scenario_path /args.env_file
    
    Bot = BOT(**vars(args))

    if args.base:
        Bot.load_agent(time=args.base)

    Bot.train()
        


        
    
