#!/usr/bin/env python3
import argparse
import sys
import os
import errno

import random
import tqdm
from collections import defaultdict

from ogb.lsc import MAG240MDataset
import numpy as np
import dgl
import torch
import logging


logging.basicConfig(
     format= '%(asctime)s %(levelname)s %(module)s : %(message)s',
     level=logging.INFO
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """Set seed for all possible random place.
    """
    logger.info("Set ALL possible random seed to %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdirs(path):
    try:
        logger.info("Creating output path: %s" % path)
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def calc_randomwalk_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    feat[i, :] = ft.sum(axis=0) / ft.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_feat_features(graph, node_ids, metapath, features, feature_dim=768, num_walkers=160, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), feature_dim), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                feat[i, :] = features[mapper[sid]].astype(np.float32).mean(axis=0)
            else:
                feat[i, :] = 0
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_topk_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                dids, cnts = np.unique(dids, return_counts=True)
                itk = np.argsort(cnts)[-topk:]
                dids, cnts = dids[itk], cnts[itk]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    ft *= cnts.reshape((-1, 1))
                    feat[i, :] = ft.sum(axis=0) / cnts.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_topk_feat_features(graph, node_ids, metapath, features, feature_dim=768, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), feature_dim), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                dids, cnts = np.unique(dids, return_counts=True)
                itk = np.argsort(cnts)[-topk:]
                dids, cnts = dids[itk], cnts[itk]
                ft = features[dids].astype(np.float32) * cnts.reshape((-1, 1))
                feat[i, :] = ft.sum(axis=0) / cnts.sum()
            else:
                feat[i, :] = 0
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_neighborsample_label_features(graph, node_ids, metapath, labels, num_classes=153):
    feat = np.zeros((len(node_ids), num_classes), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        nids = nids0
        for mp in metapath:
            _, nids = map(lambda x: x.numpy(), graph.out_edges(nids, form='uv', etype=mp))
        nids = np.unique(nids[(nids != nids0)])

        if len(nids) > 0:
            lbs = labels[nids]
            lbs = lbs[(lbs >= 0)].astype(np.int64)
            if len(lbs) == 0:
                feat[i, :] = 1. / num_classes
            else:
                ft = np.zeros((len(lbs), num_classes), dtype=np.float32)
                ft[list(range(len(lbs))), lbs] = 1
                feat[i, :] = ft.sum(axis=0) / ft.sum()
        else:
            feat[i, :] = 1. / num_classes
    return feat


def calc_neighborsample_feat_features(graph, node_ids, metapath, features, feature_dim=768):
    feat = np.zeros((len(node_ids), feature_dim), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        nids = nids0
        for mp in metapath:
            _, nids = map(lambda x: x.numpy(), graph.out_edges(nids, form='uv', etype=mp))
        nids = np.unique(nids[(nids != nids0)])

        if len(nids) > 0:
            feat[i, :] = features[nids].astype(np.float32).mean(axis=0)
        else:
            feat[i, :] = 0
    return feat


def calc_neighborsample_filter_label_features(graph, node_ids, metapath, labels, num_classes=153, ftype='least', num_common=2):
    if ftype not in {'least', 'common'}:
        raise ValueError("Unknown ftype: %r, only support 'least' and 'common'" % ftype)
    if len(metapath) != 2:
        raise ValueError("metapath should with length 2: %r" % metapath)

    feat = np.zeros((len(node_ids), num_classes), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        dids = nids0
        for mp in metapath:
            sids, dids = map(lambda x: x.numpy(), graph.out_edges(dids, form='uv', etype=mp))

        if ftype == 'least':
            nids, cnts = np.unique(sids, return_counts=True)
            sid = nids[np.argmin(cnts)]  # least middle
            nids = dids[sids == sid]
            nids = nids[(nids != nids0)]
        else:
            nids, cnts = np.unique(dids, return_counts=True)
            nids = nids[cnts >= num_common]
            nids = nids[(nids != nids0)]

        if len(nids) > 0:
            lbs = labels[nids]
            lbs = lbs[(lbs >= 0)].astype(np.int64)
            if len(lbs) == 0:
                feat[i, :] = 1. / num_classes
            else:
                ft = np.zeros((len(lbs), num_classes), dtype=np.float32)
                ft[list(range(len(lbs))), lbs] = 1
                feat[i, :] = ft.sum(axis=0) / ft.sum()
        else:
            feat[i, :] = 1. / num_classes
    return feat


def calc_neighborsample_filter_feat_features(graph, node_ids, metapath, features, feature_dim=768, ftype='least', num_common=2):
    if ftype not in {'least', 'common'}:
        raise ValueError("Unknown ftype: %r, only support 'least' and 'common'" % ftype)
    if len(metapath) != 2:
        raise ValueError("metapath should with length 2: %r" % metapath)

    feat = np.zeros((len(node_ids), feature_dim), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        dids = nids0
        for mp in metapath:
            sids, dids = map(lambda x: x.numpy(), graph.out_edges(dids, form='uv', etype=mp))

        if ftype == 'least':
            nids, cnts = np.unique(sids, return_counts=True)
            sid = nids[np.argmin(cnts)]  # least middle
            nids = dids[sids == sid]
            nids = nids[(nids != nids0)]
        else:
            nids, cnts = np.unique(dids, return_counts=True)
            nids = nids[cnts >= num_common]
            nids = nids[(nids != nids0)]

        if len(nids) > 0:
            feat[i, :] = features[nids].astype(np.float32).mean(axis=0)
        else:
            feat[i, :] = 0
    return feat


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Feature Engineering for OGB-MAG240M',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('dataset_path', help='The directory of dataset')
    parser.add_argument('graph_filename', help='The filename of input heterogeneous graph (coo or csr format)')
    parser.add_argument('output_path', help='The directory of output data')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    mkdirs(args.output_path)
    seed_everything(args.seed)

    dataset = MAG240MDataset(root=args.dataset_path)

    node_ids = np.concatenate(
        [dataset.get_idx_split(c).astype(np.int64) for c in ['train', 'valid', 'test']]
    )
    paper_feat = dataset.paper_feat
    paper_label = dataset.all_paper_label
    paper_year = dataset.all_paper_year

    feature_dim, num_classes = paper_feat.shape[1], dataset.num_classes
    logger.info("Paper feature dimension: %d, Paper class number: %d" % (feature_dim, num_classes))

    graph = dgl.load_graphs(args.graph_filename)[0][0]
    graph = graph.formats(['csr'])  # when use crc format, out_edges return incorrect result

    # 0. base
    logger.info('base')
    x_base, y_base = paper_feat[node_ids], paper_label[node_ids]
    y_base = y_base[y_base >= 0]  # get ride of test
    np.save(os.path.join(args.output_path, 'x_base.npy'), x_base)
    np.save(os.path.join(args.output_path, 'y_base.npy'), y_base)

    # 1. random walk
    metapaths = {
        'pcp': ['cites'],
        'pcbp': ['cited_by'],
        'pcpcbp': ['cites', 'cited_by'],
        'pcbpcp': ['cited_by', 'cites'],
        'pcpcp': ['cites', 'cites'],
        'pcbpcbp': ['cited_by', 'cited_by'],
        'pwbawp': ['writed_by', 'writes']
    }
    for n, mp in metapaths.items():
        logger.info(n, mp)
#         x = calc_randomwalk_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
#                                           num_walkers=160)
#         np.save(os.path.join(args.output_path, 'x_%s_rw_fmean.npy' % n), x)
        x = calc_randomwalk_label_features(graph, node_ids, mp, paper_label, num_classes,
                                           num_walkers=160)
        np.save(os.path.join(args.output_path, 'x_%s_rw_lratio.npy' % n), x)

    # 2. random walk topk
    metapaths = {
        'pcp': ['cites'],
        'pcbp': ['cited_by'],
        'pcpcbp': ['cites', 'cited_by'],
        'pcbpcp': ['cited_by', 'cites'],
        'pcpcp': ['cites', 'cites'],
        'pcbpcbp': ['cited_by', 'cited_by'],
        'pwbawp': ['writed_by', 'writes']
    }
    topk = 10
    for n, mp in metapaths.items():
        logger.info(n, mp)
#         x = calc_randomwalk_topk_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
#                                                num_walkers=160, topk=topk)
#         np.save(os.path.join(args.output_path, 'x_%s_rw_top%d_fmean.npy' % (n, topk)), x)
        x = calc_randomwalk_topk_label_features(graph, node_ids, mp, paper_label, num_classes,
                                                num_walkers=160, topk=topk)
        np.save(os.path.join(args.output_path, 'x_%s_rw_top%d_lratio.npy' % (n, topk)), x)

    # neighbor sample
    metapaths = {
        'pcp': ['cites'],
        'pcbp': ['cited_by'],
        'pcpcbp': ['cites', 'cited_by'],
        'pcbpcp': ['cited_by', 'cites'],
        'pcpcp': ['cites', 'cites'],
        'pcbpcbp': ['cited_by', 'cited_by'],
        'pwbawp': ['writed_by', 'writes']
    }
    for n, mp in metapaths.items():
        logger.info(n, mp)
#         x = calc_neighborsample_feat_features(graph, node_ids, mp, paper_feat, feature_dim)
#         np.save(os.path.join(args.output_path, 'x_%s_ns_fmean.npy' % n), x)
        x = calc_neighborsample_label_features(graph, node_ids, mp, paper_label, num_classes)
        np.save(os.path.join(args.output_path, 'x_%s_ns_lratio.npy' % n), x)

    # neighbor sample by 'least' or 'common'
    metapaths = {
        'pwbawp': ['writed_by', 'writes']
    }
    for n, mp in metapaths.items():
        logger.info(n, mp)
#         x = calc_neighborsample_filter_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
#                                                      ftype='common', num_common=2)
#         np.save(os.path.join(args.output_path, 'x_%s_ns_c2_fmean.npy' % n), x)
        x = calc_neighborsample_filter_label_features(graph, node_ids, mp, paper_label, num_classes,
                                                      ftype='common', num_common=2)
        np.save(os.path.join(args.output_path, 'x_%s_ns_c2_lratio.npy' % n), x)

#         x = calc_neighborsample_filter_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
#                                                      ftype='common', num_common=4)
#         np.save(os.path.join(args.output_path, 'x_%s_ns_c4_fmean.npy' % n), x)
        x = calc_neighborsample_filter_label_features(graph, node_ids, mp, paper_label, num_classes,
                                                      ftype='common', num_common=4)
        np.save(os.path.join(args.output_path, 'x_%s_ns_c4_lratio.npy' % n), x)

#         x = calc_neighborsample_filter_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
#                                                      ftype='least')
#         np.save(os.path.join(args.output_path, 'x_%s_ns_l_fmean.npy' % n), x)
        x = calc_neighborsample_filter_label_features(graph, node_ids, mp, paper_label, num_classes,
                                                      ftype='least')
        np.save(os.path.join(args.output_path, 'x_%s_ns_l_lratio.npy' % n), x)

    logger.info("DONE")
