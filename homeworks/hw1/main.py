from deepul.hw1_helper import *
import argparse
import sys
import homeworks.hw1.hw1_solved as hw1
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true",
                    help="Training on gpu.")

parser.add_argument("ex", type=str, choices=["q1_a", "q1_b", "q2_a", "q2_b",
                                             "q3_a", "q3_b", "q3_c", "q3_d",
                                             "q4_a", "q4_b"],
                    help="Code of hw to do.")

parser.add_argument("ds", type=int, choices=[1, 2],
                    help="Code of dataset to use.")

parser.add_argument("--reload", action="store_true",
                    help="Reload model from previously saved state.")

parser.add_argument("--notrain", action="store_true",
                    help="Do not train model.")

parser.add_argument("--short", action="store_true",
                    help="Short training of one epoch only (otherwise use the default of each exercise).")

parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for training.")

parser.add_argument("--epochs", type=int, default=15,
                    help="Max number of epochs.")

parser.add_argument("--bs", type=int, default=128,
                    help="Batch size.")

args = parser.parse_args()

hw1.DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
hw1.RELOAD = True if args.reload else False
hw1.TRAIN = False if args.notrain else True
hw1.LEARN_RATE = args.lr
hw1.MAX_EPOCHS = 1 if args.short else args.epochs
hw1.BATCH_SIZE = args.bs


origout = sys.stdout
origerr = sys.stderr
logpath = f'{resultsdir}/{args.ex}_dset{args.ds}.log'
Path(logpath).parent.mkdir(parents=True, exist_ok=True)
logfile = open(logpath, 'w')
sys.stdout = sys.err = logfile
print(f"Traning {args.ex}_{args.ds} with lr {args.lr} bs {args.bs} for {args.epochs} epochs.")

if args.ex == "q1_a":
    q1_save_results(args.ds, 'a', hw1.q1_a)
elif args.ex == "q1_b":
    q1_save_results(args.ds, 'b', hw1.q1_b)
elif args.ex == "q2_a":
    q2_save_results(args.ds, 'a', hw1.q2_a)
elif args.ex == "q2_b":
    q2_save_results(args.ds, 'b', hw1.q2_b)
elif args.ex == "q3_a":
    q3a_save_results(args.ds, hw1.q3_a)
elif args.ex == "q3_b":
    q3bc_save_results(args.ds, 'b', hw1.q3_b)
elif args.ex == "q3_c":
    q3bc_save_results(args.ds, 'c', hw1.q3_c)
elif args.ex == "q3_d":
    q3d_save_results(args.ds, hw1.q3_d)
elif args.ex == "q4_a":
    q4a_save_results(hw1.q4_a)
elif args.ex == "q4_b":
    q4b_save_results(hw1.q4_b)



sys.stdout = origout
sys.err = origerr
