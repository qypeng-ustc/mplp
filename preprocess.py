# modifed from dgl baseline (https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M)
#   1. change 5 edge types graph to 6 edge types graph
#   2. save CSR graph for feature process
import argparse
import os
import tqdm
import numpy as np
import torch
import dgl
import dgl.function as fn
from ogb.lsc import MAG240MDataset


parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, default='.',
                    help='Directory to download the OGB dataset.')
parser.add_argument('--author-output-path', type=str,
                    help='Path to store the author features.')
parser.add_argument('--inst-output-path', type=str,
                    help='Path to store the institution features.')
parser.add_argument('--graph-output-path', type=str,
                    help='Path to store the graph.')
parser.add_argument('--graph-as-homogeneous', action='store_true',
                    help='Store the graph as DGL homogeneous graph.')
parser.add_argument('--full-output-path', type=str,
                    help='Path to store features of all nodes.  Effective only when graph is homogeneous.')
args = parser.parse_args()


print('Building graph')
dataset = MAG240MDataset(root=args.rootdir)
ei_writes = dataset.edge_index('author', 'writes', 'paper')
ei_cites = dataset.edge_index('paper', 'paper')
ei_affiliated = dataset.edge_index('author', 'institution')

# We sort the nodes starting with the papers, then the authors, then the institutions. [BUG]: author, institution, paper?
author_offset = 0
inst_offset = author_offset + dataset.num_authors
paper_offset = inst_offset + dataset.num_institutions


# different from dgl baseline, here has 6 edge types.
g = dgl.heterograph({
    ('author', 'writes', 'paper'): (ei_writes[0], ei_writes[1]),
    ('paper', 'writed_by', 'author'): (ei_writes[1], ei_writes[0]),
    ('author', 'affiliated_with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
    ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
    ('paper', 'cites', 'paper'): (ei_cites[0], ei_cites[1]),
    ('paper', 'cited_by', 'paper'): (ei_cites[1], ei_cites[0])
})

dgl.save_graphs(os.path.join(args.graph_output_path, 'dgl_graph_full_heterogeneous.bin'), g)

paper_feat = dataset.paper_feat
author_feat = np.memmap(args.author_output_path, mode='w+', dtype='float16', shape=(dataset.num_authors, dataset.num_paper_features))
inst_feat = np.memmap(args.inst_output_path, mode='w+', dtype='float16', shape=(dataset.num_institutions, dataset.num_paper_features))

# Iteratively process author features along the feature dimension.
BLOCK_COLS = 16
with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
    for start in tq:
        tq.set_postfix_str('Reading paper features...')
        g.nodes['paper'].data['x'] = torch.FloatTensor(paper_feat[:, start:start + BLOCK_COLS].astype('float32'))
        # Compute author features...
        tq.set_postfix_str('Computing author features...')
        g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='writed_by')
        # Then institution features...
        tq.set_postfix_str('Computing institution features...')
        g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliated_with')
        tq.set_postfix_str('Writing author features...')
        author_feat[:, start:start + BLOCK_COLS] = g.nodes['author'].data['x'].numpy().astype('float16')
        tq.set_postfix_str('Writing institution features...')
        inst_feat[:, start:start + BLOCK_COLS] = g.nodes['institution'].data['x'].numpy().astype('float16')
        del g.nodes['paper'].data['x']
        del g.nodes['author'].data['x']
        del g.nodes['institution'].data['x']
author_feat.flush()
inst_feat.flush()


# The FEATURE needs CSR graph
g = g.formats(['csr'])
dgl.save_graphs(os.path.join(args.graph_output_path, 'dgl_graph_full_heterogeneous_csr.bin'), g)

# Function to_homogeneous may need csc format
g = g.formats(['coo'])

# Convert to homogeneous if needed.  (The RGAT baseline needs homogeneous graph)
if args.graph_as_homogeneous:
    # Process graph
    g = dgl.to_homogeneous(g)
    # DGL ensures that nodes with the same type are put together with the order preserved.
    # DGL also ensures that the node types are sorted in ascending order.
    assert torch.equal(
        g.ndata[dgl.NTYPE],
        torch.cat([torch.full((dataset.num_authors,), 0),
                   torch.full((dataset.num_institutions,), 1),
                   torch.full((dataset.num_papers,), 2)]))
    assert torch.equal(
        g.ndata[dgl.NID],
        torch.cat([torch.arange(dataset.num_authors),
                   torch.arange(dataset.num_institutions),
                   torch.arange(dataset.num_papers)]))
    g.edata['etype'] = g.edata[dgl.ETYPE].byte()
    del g.edata[dgl.ETYPE]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]

    # Process feature
    full_feat = np.memmap(
        args.full_output_path, mode='w+', dtype='float16',
        shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, dataset.num_paper_features))
    BLOCK_ROWS = 100000
    for start in tqdm.trange(0, dataset.num_authors, BLOCK_ROWS):
        end = min(dataset.num_authors, start + BLOCK_ROWS)
        full_feat[author_offset + start:author_offset + end] = author_feat[start:end]
    for start in tqdm.trange(0, dataset.num_institutions, BLOCK_ROWS):
        end = min(dataset.num_institutions, start + BLOCK_ROWS)
        full_feat[inst_offset + start:inst_offset + end] = inst_feat[start:end]
    for start in tqdm.trange(0, dataset.num_papers, BLOCK_ROWS):
        end = min(dataset.num_papers, start + BLOCK_ROWS)
        full_feat[paper_offset + start:paper_offset + end] = paper_feat[start:end]

    # Convert the graph to the given format and save.  (The RGAT baseline needs CSC graph)
    g = g.formats(['csc'])
    dgl.save_graphs(os.path.join(args.graph_output_path, 'dgl_graph_full_homogeneous_csc.bin'), g)
