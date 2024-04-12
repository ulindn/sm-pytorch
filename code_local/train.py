# References: 
## https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
## https://github.com/aws/amazon-sagemaker-examples/tree/main/frameworks/pytorch
## https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
## https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

# standard imports
import os
import argparse
import sys
import json

# local imports
import dataset
import model_def

# installed packages
import torch
from torch import nn
import logging

# enable logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_training_device(use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def test_per_epoch(device, dataloader, size, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return


def train(args):
    use_cuda = args.num_gpus > 0
    device = get_training_device(use_cuda)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    train_dataloader, test_dataloader = dataset.create_data_loaders(batch_size=args.batch_size)

    model = model_def.NeuralNetwork().to(device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    logger.info("Start training ...")
    #model.train()
    size = len(train_dataloader.dataset)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % args.log_interval == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        test_per_epoch(device, test_dataloader, size, model, loss_fn)
    
    # save model checkpoint
    save_model(model, args.model_dir)
    return



def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    print('f{Model saved in: {model_dir}}')
    return


def parse_args():
    # TBD: change to remove any unused args

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.9, metavar="BETA1", help="beta1 (default: 0.9)"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, metavar="BETA2", help="beta2 (default: 0.999)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="L2 weight decay (default: 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

