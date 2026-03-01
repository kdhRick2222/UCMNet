import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--dataset', type=str, default="TOLED", help='TOLED, POLED, SYNTH')
parser.add_argument('--epochs', type=int, default=4000, help='maximum number of epochs to train the total model.')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers.')

options = parser.parse_args()
