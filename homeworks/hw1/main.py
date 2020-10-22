from deepul.hw1_helper import *
import argparse
import sys
import homeworks.hw1.hw1_solved as hw1
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true",
                    help="Training on gpu.")

parser.add_argument("ex", type=str, choices=["q1a", "q1b", "q2a", "q2b", "q3a", "q3b", "q3c", "q3d"],
                    help="Code of hw to do.")

parser.add_argument("ds", type=int, choices=[1, 2],
                    help="Code of dataset to use.")

parser.add_argument("--reload", action="store_true",
                    help="Reload model from previously saved state.")

parser.add_argument("--notrain", action="store_true",
                    help="Do not train model.")

parser.add_argument("--short", action="store_true",
                    help="Short training of one epoch only (otherwise use the default of each exercise).")

args = parser.parse_args()

hw1.DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
hw1.RELOAD = True if args.reload else False
hw1.TRAIN = False if args.notrain else True
hw1.SHORTTRAINING = True if args.short else False


origout = sys.stdout
origerr = sys.stderr
Path(f'{resultsdir}/{args.ex}_{args.ds}.out').touch(exist_ok=True)
logfile = open(f'{resultsdir}/{args.ex}_{args.ds}.out', 'w')
sys.stdout = sys.err = logfile

if args.ex == "q1a":
    q1_save_results(args.ds, 'a', hw1.q1_a)
elif args.ex == "q1b":
    q1_save_results(args.ds, 'b', hw1.q1_b)
elif args.ex == "q2a":
    q2_save_results(args.ds, 'a', hw1.q2_a)
elif args.ex == "q2b":
    q2_save_results(args.ds, 'b', hw1.q2_b)
elif args.ex == "q3a":
    q3a_save_results(args.ds, hw1.q3_a)
elif args.ex == "q3b":
    q3bc_save_results(args.ds, 'b', hw1.q3_b)
elif args.ex == "q3c":
    q3bc_save_results(args.ds, 'c', hw1.q3_c)
elif args.ex == "q3d":
    q3d_save_results(args.ds, hw1.q3_d)

sys.stdout = origout
sys.err = origerr
