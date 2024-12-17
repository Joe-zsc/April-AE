import os
import sys

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env_file",
        default=r"single\env-CVE-2017-10271.json",
        help=r"training data set, e.g. single\env-CVE-2019-9193.json",
    )
    parser.add_argument("--agent", default="SAC_AE")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--new", type=bool, default=False, help="gpu id")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from util import set_logger, UTIL, set_seed

    set_logger(UTIL.log_path / "April-AE_train.log")
    set_seed(args.seed)
    scenario_path = UTIL.project_path / "scenarios"
    args.env_file = scenario_path / args.env_file

    from Bot import BOT

    Bot = BOT(**vars(args))
    Bot.train()
