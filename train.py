# -*- coding: utf-8 -*-
import os
import copy
import argparse
import random

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn import metrics
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support, classification_report
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils

from datasets.ABIDEDataset import ABIDEDataset
from datasets.utils import train_val_test_split,train_test_splitKFold,StratifiedKFold_tr_te_lab,StratifiedShuffleSplit_tr_vl_te,StratifiedShuffleSplit_tr_te,StratifiedKFold_tr_te

from model.models import GraphTransformer
from model.data import GraphDataset
from model.utils import count_parameters
from utils import isfloat
from model.position_encoding import POSENCODINGS
from model.gnn_layers import GNN_TYPES
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter


def load_args():
    parser = argparse.ArgumentParser(
        description='Community-Aware Graph Transformer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--root_dir', type=str, default="D:\dataset\ABIDE_I\ABIDE_pcp\cpac/filt_noglobal/cc200", help='root path')
    # parser.add_argument('--root_dir', type=str, default="D:\dataset\ABIDE_II\ABIDE_pcp\dparsf/filt_noglobal\cc200", help='root path')
    # parser.add_argument('--root_dir', type=str, default="D:/dataset/MDD/pcp/dparsf/filt_noglobal/cc200", help='root path')
    # parser.add_argument('--root_dir', type=str, default="D:/dataset/neurocon/pcp/dparsf/filt_noglobal/cc200", help='root path')
    # parser.add_argument('--root_dir', type=str, default="D:/dataset/Taowu/pcp/dparsf/filt_noglobal/cc200", help='root path')
    # parser.add_argument('--root_dir', type=str, default="D:/dataset/t+n/pcp/dparsf/filt_noglobal/cc200", help='root path')
    parser.add_argument('--dataset', type=str, default="ABIDE_I",
                        help='name of dataset')
    parser.add_argument('--atlas', type=str, default="cc200",
                        help='name of atlas')
    parser.add_argument('--FBNC', type=str, default="PCC", help='Methods for constructing functional brain networks')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=2, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=256, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=70,
                        help='number of epochs')
    parser.add_argument('--Kfold', type=int, default=10,
                        help='number of fold')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--pe', type=str, default="rw+mni", choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--pe-dim', type=int, default=33, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='D:\PyCharmProject\SAT\logs',
                        help='output path')
    parser.add_argument('--save_logs', type=bool, default=False,
                        help='Whether to save logs')
    parser.add_argument('--warmup', type=int, default=None, help="number of iterations for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--topK-edge', type=str, default="edge30", help='number of save edge')
    parser.add_argument('--use-edge-attr', action='store_true', default=True, help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=4, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='gine',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="commgnn",
                        help='Extractor type: gnn , commgnn')
    parser.add_argument('--use-dw', action='store_true', default=True, help='use distance weight')
    parser.add_argument('--use-cw', action='store_true', default=True, help='use community weight')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    if args.outdir != '' and args.save_logs == True:

        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}/{}/{}'.format(args.atlas, args.FBNC, args.topK_edge)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.pe is None else '{}_{}'.format(args.pe, args.pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )

        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args

def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    total_loss = 0
    total_correct = 0

    tic = timer()
    for i, data in enumerate(loader):
        # data.y = torch.squeeze(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        # flooding loss
        # loss = (loss - 0.15).abs() + 0.15
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == data.y).sum().item()

    toc = timer()
    n_sample = len(loader.dataset)
    avg_loss = total_loss / n_sample
    accuracy = total_correct / n_sample
    print('Train loss: {:.4f} Train acc: {:.4f} time: {:.2f}s'.format(avg_loss, accuracy, toc - tic))

    return avg_loss, accuracy


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()
    total_loss = 0
    total_correct = 0
    labels = []
    result = []

    tic = timer()
    with torch.no_grad():
        for data in loader:
            # data.y = torch.squeeze(data.y)
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)

            total_loss += loss.item() * data.num_graphs
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == data.y).sum().item()
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += data.y.tolist()
    toc = timer()
    auc = roc_auc_score(labels, result)
    result, labels = np.array(result), np.array(labels)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    metric = precision_recall_fscore_support(
        labels, result, average='macro')

    report = classification_report(
        labels, result, output_dict=True, zero_division=0)

    recall = [0, 0]
    for k in report:
        if isfloat(k):
            recall[int(float(k))] = report[k]['recall']

    n_sample = len(loader.dataset)

    avg_loss = total_loss / n_sample
    accuracy = total_correct / n_sample
    print('{} loss: {:.4f} acc: {:.4f} auc: {:.4f} time: {:.2f}s'.format(
        split, avg_loss, accuracy, auc, toc - tic))
    return avg_loss,accuracy, [auc] + list(metric) + recall



def main():
    global args
    args = load_args()
    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    print(args)

    if args.FBNC == "PCC":
        path = args.root_dir
    else:
        path = os.path.join(args.root_dir, args.data_dir)
    dataset = ABIDEDataset(path)

    tr_index, te_index = train_test_splitKFold(kfold=args.Kfold, random_state=42, n_sub=len(dataset))

    acc_list = []
    auc_list = []
    sen_list = []
    spec_list = []
    rec_list = []
    pre_list = []
    f1_list = []
    for fold in range(0,args.Kfold):
        outdir= args.outdir+'/'+str(args.Kfold)+'Fold/fold_'+str(fold+1)

        if args.save_logs:
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass


        train_loader = DataLoader(dataset[tr_index[fold]], batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset[te_index[fold]], batch_size=args.batch_size, shuffle=False, drop_last=False)


        print(dataset[tr_index[fold]][0])
        model = GraphTransformer(in_size=dataset.num_node_features,
                                 num_class=2,
                                 d_model=args.dim_hidden,
                                 dim_feedforward=2 * args.dim_hidden,
                                 dropout=args.dropout,
                                 num_heads=args.num_heads,
                                 num_layers=args.num_layers,
                                 batch_norm=args.batch_norm,
                                 pe=args.pe,
                                 pe_dim=args.pe_dim,
                                 gnn_type=args.gnn_type,
                                 use_edge_attr=args.use_edge_attr,
                                 num_edge_features=dataset.num_edge_features,
                                 edge_dim=args.edge_dim,
                                 se=args.se)

        if args.use_cuda:
            model.cuda()
        print("Total number of parameters: {}".format(count_parameters(model)))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmup is None:
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                    factor=0.5,
                                                                    patience=10,
                                                                    min_lr=1e-05,
                                                                    verbose=False)
        else:
            lr_steps = (args.lr - 1e-6) / args.warmup
            decay_factor = args.lr * args.warmup ** .5

            def lr_scheduler(s):
                if s < args.warmup:
                    lr = 1e-6 + s * lr_steps
                else:
                    lr = decay_factor * s ** -.5
                return lr

        print("Training... fold:{}".format(fold+1))

        best_test_acc = 0
        best_model = None
        best_epoch = 0
        logs = defaultdict(list)
        start_time = timer()
        for epoch in range(args.epochs):
            print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)

            test_loss, test_acc, test_result = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

            if args.warmup is None:
                lr_scheduler.step(test_acc)

            if args.save_logs:
                logs['train_loss'].append(train_loss)
                logs['train_acc'].append(train_acc)
                logs['test_loss'].append(test_loss)
                logs['test_acc'].append(test_acc)
                tag_scalar_dict_loss = {
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                }
                tag_scalar_dict_acc = {
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }
                tag_scalar_dict_auc = {
                    'test_auc': test_result[0]
                }
                tag_scalar_dict_sen = {
                    'test_sen': test_result[-1]
                }
                tag_scalar_dict_spec = {
                    'test_spec': test_result[-2]
                }
                writer = SummaryWriter(outdir + '/tensorboard_log')
                writer.add_scalars(main_tag='loss', tag_scalar_dict=tag_scalar_dict_loss, global_step=epoch)
                writer.add_scalars(main_tag='acc', tag_scalar_dict=tag_scalar_dict_acc, global_step=epoch)
                writer.add_scalars(main_tag='auc', tag_scalar_dict=tag_scalar_dict_auc, global_step=epoch)
                writer.add_scalars(main_tag='sen', tag_scalar_dict=tag_scalar_dict_sen, global_step=epoch)
                writer.add_scalars(main_tag='spec', tag_scalar_dict=tag_scalar_dict_spec, global_step=epoch)
                writer.close()

            if best_test_acc <= test_acc and epoch > 5:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model.state_dict())


        total_time = timer() - start_time
        print()
        print("best epoch: {} best test acc: {:.4f}".format(best_epoch, best_test_acc))

        model.load_state_dict(best_model)

        print("Testing...")
        test_loss, test_acc, test_result = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

        if args.save_logs:
            logs = pd.DataFrame.from_dict(logs)
            logs.to_csv(outdir + '/logs.csv')
            results = {
                'Test Loss': test_loss,
                'Best Test Accuracy': best_test_acc,
                'Test AUC': test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                'best_epoch': best_epoch,
                'total_time': total_time,
            }
            results = pd.DataFrame.from_dict(results, orient='index')
            results.to_csv(outdir + '/results.csv',
                           header=['value'], index_label='name')
            torch.save(
                {'args': args,
                 'state_dict': best_model},
                outdir + '/model.pth')
        acc_list.append(best_test_acc)
        auc_list.append(test_result[0])
        sen_list.append(test_result[-1])
        spec_list.append(test_result[-2])
        rec_list.append(test_result[-5])
        pre_list.append(test_result[-6])
        f1_list.append(test_result[-4])


    print(acc_list)
    print("test acc mean {} std {}".format(np.mean(acc_list), np.std(acc_list)))
    print("test auc mean {} std {}".format(np.mean(auc_list) * 100, np.std(auc_list) * 100))
    print("test sensitivity mean {} std {}".format(np.mean(sen_list) * 100, np.std(sen_list) * 100))
    print("test specficity mean {} std {}".format(np.mean(spec_list) * 100, np.std(spec_list) * 100))
    print("test recall mean {} std {}".format(np.mean(rec_list) * 100, np.std(rec_list) * 100))
    print("test precision mean {} std {}".format(np.mean(pre_list) * 100, np.std(pre_list) * 100))
    print("test f1_micro mean {} std {}".format(np.mean(f1_list) * 100, np.std(f1_list) * 100))


if __name__ == "__main__":
    main()

