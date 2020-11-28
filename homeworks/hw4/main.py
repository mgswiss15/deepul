from deepul.hw4_helper import *
import argparse
import homeworks.hw4.hw4_solved as hw
import wandb


parser = argparse.ArgumentParser()

# exercise specs
parser.add_argument("ex", type=str, choices=["q1_a", "q1_b", "q2"],
                    help="Code of hw to do.")

parser.add_argument("--ds", type=int, choices=[1, 2],
                    help="Code of dataset to use.")

# run specs
parser.add_argument("--gpu", action="store_true",
                    help="Training on gpu.")

parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for training.")

parser.add_argument("--epochs", type=int, default=15,
                    help="Max number of epochs.")

parser.add_argument("--bs", type=int, default=128,
                    help="Batch size.")

# arch specs
parser.add_argument("--dsteps", type=int, default=1,
                    help="Nubmer of discriminator steps per one generator step.")

parser.add_argument("--dhidden", type=int, nargs="+", default=[10, 10],
                    help="Hidden layers of the discriminator.")

parser.add_argument("--ghidden", type=int, nargs="+", default=[10, 10],
                    help="Hidden layers of the generator.")

# wandb
parser.add_argument("--wandboff", action="store_true",
                    help="Switch off writing to wandb cloud.")
parser.add_argument("--genfreq", type=int, default=50,
                    help="Frequency for plotting generations.")

args = parser.parse_args()

# If you don't want your script to sync to the cloud
if args.wandboff:
    os.environ['WANDB_MODE'] = 'dryrun'

wandb.init(project=f"deepul-hw4-{args.ex}")
wandb.config.update(args)
os.environ['WANDB_DIR'] = '../hw4/wandb'
os.environ['WANDB_SHOW_RUN'] = 'true'

hw.DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
hw.LEARN_RATE = args.lr
hw.MAX_EPOCHS = args.epochs
hw.BATCH_SIZE = args.bs
hw.GENFREQ = args.genfreq
hw.DSTEPS = args.dsteps
hw.DHIDDEN = args.dhidden
hw.GHIDDEN = args.ghidden

print(f"Traning {args.ex}_{args.ds} with lr {hw.LEARN_RATE} bs {hw.BATCH_SIZE} for {hw.MAX_EPOCHS} epochs.")

if args.ex == "q1_a":
    q1_save_results('a', hw.q1_a)
elif args.ex == "q1_b":
    q1_save_results('b', hw.q1_b)
elif args.ex == "q2":
    q2_save_results(hw.q2)
