#!/usr/bin/env python3
import argparse
import os
import errno

import random
import numpy as np

from sklearn.model_selection import KFold
import torch
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

import logging


logging.basicConfig(
     format= '%(asctime)s %(levelname)s %(module)s : %(message)s',
     level=logging.INFO
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """Set seed for all possible random place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdirs(path):
    try:
        print("Creating output path: %s" % path)
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, dropout: float = 0.0, batch_norm: bool = True,
                 relu_last: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MPLP(torch.nn.Module):
    def __init__(self, in_channels: tuple, hidden_channels: tuple, out_channels: int, num_layers: int,
                 dropout: float = 0.0, batch_norm: bool = True, relu_last: bool = False):
        super(MPLP, self).__init__()

        self.mlps = ModuleList()
        tot_hidden = 0
        for i, j in zip(in_channels, hidden_channels):
            tot_hidden += j
            mlp = MLP(i, j, j, num_layers, dropout,
                      batch_norm, relu_last)
            self.mlps.append(mlp)

        self.mlp = MLP(tot_hidden, 512, out_channels, num_layers, dropout, batch_norm, relu_last)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
        self.reset_parameters()

    def forward(self, xs):
        out = []
        for x, mlp in zip(xs, self.mlps):
            out.append(mlp(x))
        out = torch.cat(out, dim=-1).relu_()
        return self.mlp(out)


def load_data(datapath, feat_info, device=None):
    logger.info("Loading data from %s" % datapath)

    x_all = []
    for i, (fn, _, _) in enumerate(feat_info):
        logger.info("Loading features for %d: %s ..." % (i, fn.upper()))
        feat = []
        fname = os.path.join(datapath, '%s.npy' % fn)
        logger.info("Loading %s" % fname)
        feat = torch.from_numpy(np.load(fname)).to(device, torch.float32)
        x_all.append(feat)

    logger.info("Loading labels ...")
    fname = os.path.join(datapath, 'y_base.npy')
    logger.info("Loading %s" % fname)
    y_all = torch.from_numpy(np.load(fname)).to(device, torch.long)

    return x_all, y_all


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='OGB-MAG240M with MPLP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('dataset_path', help='The directory of dataset')
    parser.add_argument('input_path', help='The directory of input data')
    parser.add_argument('output_path', help='The directory of output data')

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    mkdirs(args.output_path)
    seed_everything(args.seed)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logger.info(device)

    dataset = MAG240MDataset(args.dataset_path)
    evaluator = MAG240MEvaluator()
    num_feats = dataset.num_paper_features

    feat_info = [
        ('x_rgat_1024', 1024, 128),
        ('x_base', 768, 128),
        ('x_pcbpcp_rw_lratio', 153, 32),
        ('x_pcbpcp_rw_top10_lratio', 153, 32),
        ('x_pcp_rw_lratio', 153, 32),
        ('x_pcpcbp_rw_lratio', 153, 32),
        ('x_pcpcbp_rw_top10_lratio', 153, 32),
        ('x_pcpcp_rw_lratio', 153, 32),
        ('x_pcpcp_rw_top10_lratio', 153, 32),
        ('x_pwbawp_rw_lratio', 153, 32),
        ('x_pwbawp_rw_top10_lratio', 153, 32),
        ('x_pwbawp_ns_l_lratio', 153, 32),
        ('x_pwbawp_ns_c2_lratio', 153, 32),
        ('x_pwbawp_ns_c4_lratio', 153, 32)
    ]

    logger.info("A Total of %d different type features" % len(feat_info))
    logger.info(feat_info)

    x_all, y_all = load_data(args.input_path, feat_info, device)
    logger.info("<=2019: %d, >=2020: %d" % (y_all.shape[0], x_all[0].shape[0] - y_all.shape[0]))

    split_nids = dataset.get_idx_split()

    node_ids = np.concatenate([split_nids['train'], split_nids['valid'], split_nids['test']])
    year_all = dataset.paper_year[node_ids].astype(np.int32)

    ntra, nval, ntes = [len(split_nids[c]) for c in ['train', 'valid', 'test']]

    train_idx0 = np.arange(ntra)
    valid_idx0 = np.arange(ntra, ntra + nval)
    test_idx = np.arange(ntra + nval, ntra + nval + ntes)

    # w = log10(cnt_2018 / cnt_1990_2018 + 5)
    # w = w * 153 / w.sum()
    weight = torch.FloatTensor([
       1.02510359, 1.001055  , 1.39033493, 1.01745989, 0.96811023,
       0.97076595, 0.97663513, 1.81222558, 0.97540636, 0.98365435,
       0.97666354, 0.96406336, 0.97629103, 0.97590207, 0.96573454,
       0.96727996, 1.00178323, 0.98797382, 0.9764376 , 0.98097944,
       0.9666215 , 0.9631094 , 0.96017785, 0.99481758, 0.978868  ,
       0.97575488, 0.98967411, 0.97400016, 0.98650667, 0.98130233,
       0.97339904, 0.97032673, 0.97608486, 0.9726031 , 0.96567107,
       0.98502011, 0.98676427, 0.97414173, 0.98577156, 0.99945179,
       0.96860036, 0.97388609, 0.97512986, 0.96543923, 0.97501794,
       0.97610742, 0.96692822, 0.97572049, 0.97706001, 0.96984168,
       0.96834199, 0.97358922, 0.98540044, 0.9639287 , 0.96446345,
       0.96586354, 0.97154801, 0.96764774, 0.99274776, 0.99452589,
       0.98176971, 0.96978854, 0.97026918, 0.99876552, 0.97311927,
       0.97566764, 0.97390432, 0.97075458, 0.97212568, 0.97788947,
       0.97225216, 0.99135821, 0.98841234, 1.00203906, 0.97644653,
       0.97433231, 1.32048703, 0.97576994, 0.97246489, 0.98385479,
       0.97294049, 0.97101016, 0.96750736, 0.979035  , 0.9821505 ,
       1.01540695, 0.97821276, 0.97065061, 0.98015278, 0.99141044,
       0.96469188, 0.98007539, 0.98351303, 0.96900566, 0.97789049,
       0.97191702, 0.96817653, 0.97363574, 0.97313867, 1.48244284,
       0.9601672 , 1.54390969, 0.96947606, 0.97946194, 0.97629663,
       0.9795156 , 0.97264544, 0.96848601, 0.97817272, 1.02287096,
       0.97037257, 0.97036294, 0.97611873, 0.9760138 , 0.98854796,
       0.97149293, 0.97617129, 0.97708072, 0.96581287, 0.97853862,
       0.98117139, 0.97462655, 0.98825537, 0.97237278, 0.99585226,
       0.96893638, 0.97020584, 1.17611472, 0.97347582, 0.96688439,
       0.97814826, 0.97449159, 0.96991003, 0.97936285, 0.97863592,
       0.99359083, 0.98621957, 0.98115282, 0.97841335, 0.98092584,
       1.0283728 , 0.97126237, 0.97559931, 0.9697076 , 0.99055911,
       0.9799765 , 0.96679984, 0.97838971, 0.99626964, 0.97550475,
       1.4391001 , 0.97274099, 0.98530614
    ]).to(device)

    cv = KFold(args.num_splits, shuffle=True, random_state=args.seed)

    for k, (ti, vi) in enumerate(cv.split(valid_idx0)):

        output_path = os.path.join(args.output_path, 'cv-%d' % k)
        mkdirs(output_path)

        train_idx = np.concatenate([train_idx0, valid_idx0[ti]])
        valid_idx = valid_idx0[vi]
        logger.info("KFold: %d, Train: %d, Valid: %d" % (k, len(train_idx), len(valid_idx)))

        model = MPLP(
            [fi[1] for fi in feat_info], [fi[2] for fi in feat_info],
            dataset.num_classes,
            args.num_layers, args.dropout, not args.no_batch_norm, args.relu_last
        )
        if k == 0:
            logger.info(model)
            logger.info('#Params: %d' % sum([p.numel() for p in model.parameters()]))
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)

        best_valid_acc = 0
        for epoch in range(1, 1 + args.epochs):

            model.train()
            total_loss = 0
            for idx in DataLoader(train_idx, args.batch_size, shuffle=True):
                optimizer.zero_grad()
                y_pred = model([x[idx].to(torch.float32) for x in x_all])
                y_true = y_all[idx]
                loss = F.cross_entropy(y_pred, y_true, weight=weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.numel()
            loss = total_loss / train_idx.shape[0]

            with torch.no_grad():
                model.eval()
                y_pred = []
                for idx in DataLoader(train_idx, args.batch_size):
                    y = model([x[idx].to(torch.float32) for x in x_all])
                    y_pred.append(y.argmax(dim=-1).cpu())
                y_pred = torch.cat(y_pred, dim=-1)
                train_acc = evaluator.eval({'y_true': y_all[train_idx].cpu(), 'y_pred': y_pred})['acc']

            with torch.no_grad():
                model.eval()
                y_pred = []
                for idx in DataLoader(valid_idx, args.batch_size):
                    y = model([x[idx].to(torch.float32) for x in x_all])
                    y_pred.append(y.argmax(dim=-1).cpu())
                y_pred = torch.cat(y_pred, dim=-1)
            valid_acc = evaluator.eval({'y_true': y_all[valid_idx].cpu(), 'y_pred': y_pred})['acc']

            if valid_acc > best_valid_acc:
                torch.save(model.state_dict(), os.path.join(output_path, 'model.pt'))
                best_valid_acc = valid_acc

            logger.info('Epoch: %03d, lr: %.7f, Loss: %.4f, Train: %.4f, Valid: %.4f, Best: %.4f' %
                        (epoch, sched.get_last_lr()[0], loss, train_acc, valid_acc, best_valid_acc))

            sched.step()

        logger.info("Finetune the model using latest data ...")

        model.load_state_dict(torch.load(os.path.join(output_path, 'model.pt')))
        model = model.to(device)

        train_idx_ft = train_idx[year_all[train_idx] >= 2018]
        logger.info("Train: %d, Valid: %d" % (len(train_idx_ft), len(valid_idx)))

        best_valid_acc = 0
        for epoch in range(1, 30):

            model.train()
            total_loss = 0
            for idx in DataLoader(train_idx_ft, args.batch_size, shuffle=True):
                optimizer.zero_grad()
                y_pred = model([x[idx].to(torch.float32) for x in x_all])
                y_true = y_all[idx]
                loss = F.cross_entropy(y_pred, y_true, weight=weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.numel()
            loss = total_loss / train_idx_ft.shape[0]

            with torch.no_grad():
                model.eval()
                y_pred = []
                for idx in DataLoader(train_idx_ft, args.batch_size):
                    y = model([x[idx].to(torch.float32) for x in x_all])
                    y_pred.append(y.argmax(dim=-1).cpu())
                y_pred = torch.cat(y_pred, dim=-1)
                train_acc = evaluator.eval({'y_true': y_all[train_idx_ft].cpu(), 'y_pred': y_pred})['acc']

            with torch.no_grad():
                model.eval()
                y_pred = []
                for idx in DataLoader(valid_idx, args.batch_size):
                    y = model([x[idx].to(torch.float32) for x in x_all])
                    y_pred.append(y.argmax(dim=-1).cpu())
                y_pred = torch.cat(y_pred, dim=-1)
            valid_acc = evaluator.eval({'y_true': y_all[valid_idx].cpu(), 'y_pred': y_pred})['acc']

            if valid_acc > best_valid_acc:
                torch.save(model.state_dict(), os.path.join(output_path, 'model-finetune.pt'))
                best_valid_acc = valid_acc

            logger.info('Epoch: %03d, lr: %.7f, Loss: %.4f, Train: %.4f, Valid: %.4f, Best: %.4f' %
                        (epoch, sched.get_last_lr()[0], loss, train_acc, valid_acc, best_valid_acc))

        logger.info('Predicting for train/valid/test PAPER nodes...')

        model.load_state_dict(torch.load(os.path.join(output_path, 'model-finetune.pt')))
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            model.eval()
            y_pred = []
            for idx in DataLoader(train_idx, args.batch_size):
                y = model([x[idx].to(torch.float32) for x in x_all])
                y_pred.append(y.cpu().numpy())
            y_pred = np.concatenate(y_pred, axis=0)
            np.save(os.path.join(output_path, "idx_train.npy"), train_idx)
            np.save(os.path.join(output_path, "y_pred_train.npy"), y_pred)

        with torch.no_grad():
            model.eval()
            y_pred = []
            for idx in DataLoader(valid_idx, args.batch_size):
                y = model([x[idx].to(torch.float32) for x in x_all])
                y_pred.append(y.cpu().numpy())
            y_pred = np.concatenate(y_pred, axis=0)
            np.save(os.path.join(output_path, "idx_valid.npy"), valid_idx)
            np.save(os.path.join(output_path, "y_pred_valid.npy"), y_pred)

        with torch.no_grad():
            model.eval()
            y_pred = []
            for idx in DataLoader(test_idx, args.batch_size):
                y = model([x[idx].to(torch.float32) for x in x_all])
                y_pred.append(y.cpu().numpy())
            y_pred = np.concatenate(y_pred, axis=0)
            np.save(os.path.join(output_path, "y_pred_test.npy"), y_pred)

        logger.info('Saving test PAPER prediction to %s' % output_path)
        res = {'y_pred': y_pred.argmax(axis=1)}
        evaluator.save_test_submission(res, output_path)

    logger.info('DONE!')
