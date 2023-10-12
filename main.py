import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
from torch.utils.tensorboard import SummaryWriter

import os
import time
import random
import argparse

from model.fsnet import FSNet
from utils.loss import classification_loss, reconstruction_loss
from utils.dataset import DatasetFromJson, DatasetForImbalance, get_metadata, collate_fn
from utils.common import Logger, gen_model_name


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--bsz', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='loss balancing factor')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument('--data', type=str,
                        help='dataset directory')
    parser.add_argument('--save', type=str, default='./save/',
                        help='model save directory')
    parser.add_argument('--log', type=str, default='./log/',
                        help='log directory')
    parser.add_argument('--tensorboard', type=str, default='./tensorboard/',
                        help='tensorboard directory')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of dataloader workers')

    return parser.parse_args()


args = get_args()
model_name = gen_model_name(args)

Logger.config(args, model_name)
logger = Logger(__file__)

tb_path = os.path.join(args.tensorboard, model_name)
writer = SummaryWriter(tb_path)

# random.seed(11)
# np.random.seed(22)
# torch.manual_seed(33)

train_path = os.path.join(args.data, 'train.json')
test_path = os.path.join(args.data, 'test.json')
meta_path = os.path.join(args.data, 'meta.json')

metadata = get_metadata(meta_path)
train_dataset = DatasetForImbalance('train', metadata, train_path)
test_dataset = DatasetForImbalance('test', metadata, test_path)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.bsz,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=args.workers
)


def train(epoch):
    model.train()
    total_loss = np.zeros(3)
    interval_loss = np.zeros(3)
    start_time = time.time()
    log_interval = 20
    nsample = len(train_dataset)
    nbatch = nsample // train_loader.batch_size

    for i, sample in enumerate(train_loader):
        label = sample['label'].cuda()
        seq = [seq.cuda() for seq in sample['seq']]

        # print('label', label)
        # print('seq', seq)

        if epoch == 0 and i == 0:
            writer.add_graph(model, input_to_model=seq[0], verbose=False)
            writer.close()

        app_prob, value_prob = model(seq)

        clf_loss = classification_loss(app_prob, label)
        rec_loss = reconstruction_loss(value_prob, seq)
        loss = clf_loss + args.alpha * rec_loss

        # print('clf_loss', clf_loss.item())
        # print('rec_loss', rec_loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        this_loss = np.array([clf_loss.item(), rec_loss.item(), loss.item()])
        total_loss += this_loss
        interval_loss += this_loss

        if i != 0 and i % log_interval == 0:
            cur_loss = interval_loss / log_interval
            elapsed = time.time() - start_time
            logger.info(f'epoch {epoch:3d} | '
                        f'batches {i:5d}/{nbatch:5d} | '
                        f'lr {scheduler.get_last_lr()[0]:8.1e} | '
                        f'ms/batch {elapsed * 1000 / log_interval:8.2f} | '
                        f'clf_loss {cur_loss[0]:7.4f} | '
                        f'rec_loss {cur_loss[1]:7.4f} | '
                        f'loss {cur_loss[2]:7.4f}')
            interval_loss = np.zeros(3)
            start_time = time.time()

    scheduler.step()

    return total_loss / nbatch


def pred(dataset: Dataset):
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers
    )

    model.eval()
    total_loss = np.zeros(3)
    correct = 0
    nsample = len(dataset)
    nbatch = nsample // loader.batch_size

    for i, sample in enumerate(loader):
        label = sample['label'].cuda()
        seq = [seq.cuda() for seq in sample['seq']]

        app_prob, value_prob = model(seq)
        clf_loss = classification_loss(app_prob, label)
        rec_loss = reconstruction_loss(value_prob, seq)
        loss = clf_loss + args.alpha * rec_loss

        this_loss = np.array([clf_loss.item(), rec_loss.item(), loss.item()])
        total_loss += this_loss
        correct += (app_prob.argmax(dim=1) == label).sum().item()

        # print('app_prob', app_prob, app_prob.shape)
        # print('pred', app_prob.argmax(dim=1), app_prob.argmax(dim=1).shape)
        # print('label', label, label.shape)
        # print('sum', (app_prob.argmax(dim=1) == label).sum().item(),
        #       'bsz', loader.batch_size)

        if i % 20 == 0:
            logger.info(f'Testing ... {i:5d}/{nbatch:5d}')

    avg_loss = total_loss / nbatch
    accuracy = correct / nsample
    return avg_loss, accuracy


def eval(epoch):
    test_loss, test_accuracy = pred(test_dataset)
    logger.info(f'eval test_set at epoch {epoch:3d} | '
                f'clf_loss {test_loss[0]:7.4f} | '
                f'rec_loss {test_loss[1]:7.4f} | '
                f'loss {test_loss[2]:7.4f} | '
                f'accuracy {test_accuracy:5.2f}')
    logger.scalars('test_eval', [epoch, *test_loss.tolist(), test_accuracy])

    indices = np.random.choice(
        len(train_dataset), size=len(test_dataset), replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loss, train_accuracy = pred(train_subset)
    logger.info(f'eval train_set at epoch {epoch:3d} | '
                f'clf_loss {train_loss[0]:7.4f} | '
                f'rec_loss {train_loss[1]:7.4f} | '
                f'loss {train_loss[2]:7.4f} | '
                f'accuracy {train_accuracy:5.2f}')
    logger.scalars('train_eval', [epoch, *train_loss.tolist(), train_accuracy])


with torch.cuda.device(args.gpu):
    model = FSNet(
        n_app=metadata['n_app'],
        max_value=5000,
        emb_dim=16,
        n_layer=2,
        hidden_size=128,
        dropout=0.2
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # all_count = 1 / torch.tensor(metadata['all_count'])
    # class_weights = (all_count / all_count.sum()).cuda()
    # print('class_weights', class_weights)

    logger.info(f'Start training: epochs = {args.epochs}')

    for epoch in range(args.epochs):
        start_time = time.time()
        loss = train(epoch)
        elapsed = time.time() - start_time

        logger.info(f'end of epoch {epoch:3d} | '
                    f'time {elapsed:5.2f}s | '
                    f'clf_loss {loss[0]:7.4f} | '
                    f'rec_loss {loss[1]:7.4f} | '
                    f'loss {loss[2]:7.4f}')
        logger.scalars('train_batch', [epoch, *loss.tolist()])

        if epoch % 1 == 0:  # TODO: modify eval interval
            eval(epoch)

        if not np.isnan(loss[2]):
            logger.info('Saving model ...')
            torch.save(
                model.state_dict(),
                os.path.join(args.save, f'{model_name}_checkpoint.pt'))
            if epoch % 50 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save, f'{model_name}_epoch{epoch}.pt'))
