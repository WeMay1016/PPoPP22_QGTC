import argparse
import time
import random
import os.path as osp
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
#from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import register_data_args

from modules import *
from sampler import ClusterIter
from utils import load_data, evaluate
from dataset import *
from tqdm import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
register_data_args(parser)
parser.add_argument("--gpu", type=int, default=0, help="gpu")

parser.add_argument("--n-epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=20, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-8, help="weight decay")
parser.add_argument("--psize", type=int, default=1500, help="number of partitions")

parser.add_argument("--dim", type=int, default=10, help="input dimension of each dataset")
parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
parser.add_argument("--n-classes", type=int, default=10, help="number of classes")
parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")

""" parser.add_argument("--use-pp", action='store_true',help="whether to use precomputation")
parser.add_argument("--regular", action='store_true',help="whether to use DGL")
parser.add_argument("--run_GIN", action='store_true',help="whether to run GIN model")
parser.add_argument("--use_QGTC", action='store_true',help="whether to use QGTC")
parser.add_argument("--zerotile_jump", action='store_true',help="whether to profile zero-tile jumping") """
parser.add_argument("--use-pp", type=bool, default = False, help="whether to use precomputation")
parser.add_argument("--regular", type=bool, default = True, help="whether to use DGL")
parser.add_argument("--run_GIN", type=bool, default = False, help="whether to run GIN model")
parser.add_argument("--use_QGTC", type=bool, default = False, help="whether to use QGTC")
parser.add_argument("--zerotile_jump", type=bool, default = False, help="whether to profile zero-tile jumping")

parser.add_argument("--step-size", type=int, default=10, help="schedule step")
parser.add_argument("--gamma", type=float, default=0.8, help="schedule gamma")

parser.set_defaults(dataset = 'ppi')

args = parser.parse_args()
print(args)


import torch.onnx 

#Function to Convert to ONNX 
def Convert_ONNX(model, cluster): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    #dummy_input = torch.randn(1, input_size, requires_grad=True)  
    
    name = args.dataset + '_'
    if args.run_GIN:
        name += 'GIN'
    else:
        name += 'GCN'
        
    name += '.onnx'

    # Export the model   
    torch.onnx.export(model,         # model being run 
         (cluster, cluster.ndata['feat']),       # model input (or a tuple for multiple inputs) 
         name,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')


def main(args):
    torch.manual_seed(3)
    np.random.seed(2)
    random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and preprocess dataset
    if args.dataset in ['ppi', 'reddit']:
        data = load_data(args)
        g = data.g
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        labels = g.ndata['label']
        if args.dataset == 'ppi':
            multitask = True
        else:
            multitask = False
    elif args.dataset in ['ogbn-arxiv', 'ogbn-products']:
        pass
        """ data = DglNodePropPredDataset(name=args.dataset) #'ogbn-proteins'
        split_idx = data.get_idx_split()
        g, labels = data[0]
        train_mask = split_idx['train']
        val_mask = split_idx['valid']
        test_mask = split_idx['test'] """
    else:
        path = osp.join("./qgtc_graphs", args.dataset+".npz")
        data = QGTC_dataset(path, args.dim, args.n_classes)
        g = data.g
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
    in_feats = g.ndata['feat'].shape[1]
    print(g.ndata['feat'].shape)
    n_classes = data.num_classes
    # metis only support int64 graph
    g = g.long()
    # get the subgraph based on the partitioning nodes list.
    cluster_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, train_nid, use_pp=False, regular=args.regular)

    torch.cuda.set_device(args.gpu)
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    g = g.int().to(args.gpu)

    if args.run_GIN:
        model = GIN(in_feats, args.n_hidden, n_classes)
    else:
        model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers)

    model.cuda()
    train_nid = torch.from_numpy(train_nid).cuda()
    start_time = time.time()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    
    loss_history = []
    val_acc_history = []
    model.train()
    for epoch in tqdm(range(args.n_epochs)):
        total_loss = 0
        count = 0
        for j, cluster in enumerate(cluster_iterator):
            #print(cluster.ndata['label'])
            cluster = cluster.to(torch.cuda.current_device())
            logits = model(cluster, cluster.ndata['feat'])
            train_y = cluster.ndata['label']
            if multitask == False:
                train_y = F.one_hot(train_y, num_classes=n_classes).float()
            #print(logits.shape, train_y.shape)
            #print(train_y)
            loss = criterion(logits, train_y)    # 计算损失值
            
            total_loss += loss.item()
            count += 1
            
            optimizer.zero_grad()
            loss.backward()     # 反向传播计算参数的梯度
            optimizer.step()    # 使用优化方法进行梯度更新
        
        
        #cluster = cluster.cpu()
        StepLR.step()    
        train_F1_mic, train_F1_mac, train_acc = evaluate(model, g, labels, train_mask, multitask)
        val_F1_mic, val_F1_mac, val_acc = evaluate(model, g, labels, val_mask, multitask)
        # 记录训练过程中损失值和F1_score的变化，用于画图
        avg_loss = total_loss / count
        loss_history.append(avg_loss)
        val_acc_history.append([val_F1_mic, val_F1_mac])
        
        if multitask:
            print("Epoch {:03d}: Loss {:.4f}, train_F1_mic {:.4f}, train_F1_mac {:.4f}, val_F1_mic {:.4f}, val_F1_mac {:.4f}" \
                .format(epoch, avg_loss, train_F1_mic, train_F1_mac, val_F1_mic, val_F1_mac))  
        else:
            print("Epoch {:03d}: Loss {:.4f}, train_F1_mic {:.4f}, train_F1_mac {:.4f}, train_acc {:.4f}, val_F1_mic {:.4f}, val_F1_mac {:.4f}, val_acc {:.4f}" \
                .format(epoch, avg_loss, train_F1_mic, train_F1_mac, train_acc, val_F1_mic, val_F1_mac, val_acc))  
        
    torch.cuda.synchronize()
    end_time = time.time()
    print("Avg. Epoch: {:.3f} ms".format((end_time - start_time)*1000/args.n_epochs))
    
    #模型存储
    path = './trained_model/'
    name = args.dataset + '_'
    if args.run_GIN:
        name += 'GIN'
    else:
        name += 'GCN'
        
    name += '.pt'
    torch.save(model, path + name)
    #
    
    #onnx ?
    #Convert_ONNX(model, args.batch_size, in_feats)
    #Convert_ONNX(model, cluster)

if __name__ == '__main__':
    main(args)
