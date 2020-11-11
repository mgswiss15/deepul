from deepul.hw3_helper import *
import argparse
import sys
import homeworks.hw3.hw3_solved as hw
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("ex", type=str, choices=["q1_a"],
                    help="Code of hw to do.")

parser.add_argument("--ds", type=int, choices=[1, 2],
                    help="Code of dataset to use.")

parser.add_argument("--gpu", action="store_true",
                    help="Training on gpu.")

parser.add_argument("--reload", action="store_true",
                    help="Reload model from previously saved state.")

parser.add_argument("--notrain", action="store_true",
                    help="Do not train model.")

parser.add_argument("--short", action="store_true",
                    help="Short training of one epoch only (otherwise use the default of each exercise).")

parser.add_argument("--screen", action="store_true",
                    help="Print stdout and stderr to screen (instead of log file in results folder).")

parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for training.")

parser.add_argument("--lrmax", type=float, default=0.001,
                    help="Max learning rate for training.")

parser.add_argument("--epochs", type=int, default=15,
                    help="Max number of epochs.")

parser.add_argument("--bs", type=int, default=128,
                    help="Batch size.")

args = parser.parse_args()

hw.DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
hw.RELOAD = True if args.reload else False
hw.TRAIN = False if args.notrain else True
hw.LEARN_RATE = args.lr
hw.MAXLEARN_RATE = args.lrmax
hw.MAX_EPOCHS = 1 if args.short else args.epochs
hw.BATCH_SIZE = args.bs

if not args.screen:
    origout = sys.stdout
    origerr = sys.stderr
    logpath = f'{resultsdir}/{args.ex}_dset{args.ds}.log'
    Path(logpath).parent.mkdir(parents=True, exist_ok=True)
    logfile = open(logpath, 'w')
    sys.stdout = sys.err = logfile

print(f"Traning {args.ex}_{args.ds} with lr {args.lr} bs {args.bs} for {args.epochs} epochs.")

if args.ex == "q1_a":
    q1_save_results('a', args.ds, hw.q1)

if not args.screen:
    sys.stdout = origout
    sys.err = origerr
