from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import time
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

from config import parser
from eval_metrics import recall_at_k
from models.model import TaxoRec
from geoopt.optim import RiemannianSGD
from torch import optim
from utils.data_generator import Data
from utils.helper import default_device, set_seed
from utils.sampler import WarpSampler
from utils.taxogen import build_tree_gen
import itertools, heapq
import torch
import json
torch.set_printoptions(profile='full')
import logging

def load_neighbor(dataset, order, version):
    df = pd.read_csv('/home/featurize/work/Hyper_CDR/data/'+dataset+'/neighbor_{0}_{1}.csv'.format(order, version))
    return df

def load_neighbor_dict(dataset, order):
    file_path = '/home/featurize/work/Hyper_CDR/data/' + dataset + '/neighbor_{}_dict.json'.format(order)
    with open(file_path, 'r') as file:
        neighbor_indices_dict = json.load(file)
    return neighbor_indices_dict



def train(model_tgt, data_tgt, sampler_tgt, data_drop_tgt):
    model = model_tgt
    dataset = args.dataset_tgt
    data = data_tgt
    sampler = sampler_tgt
    data_drop = data_drop_tgt
    
    neighbor_neg = load_neighbor("Amazon-Movie", 3, 'a')
    # neighbor_neg_b = load_neighbor("Amazon-CD", 3, version=b)
    # neighbor_neg = load_neighbor_dict("Amazon-Movie", 3)
    neighbor_tgt = load_neighbor_dict("Amazon-CD", 1)
    neighbor_src = load_neighbor_dict("Amazon-Movie", 1)
    
    # use_item_cl_loss = args.use_item_cl_loss
    use_user_cl_loss = args.use_user_cl_loss
    
    save_path = 'data/' + args.dataset_tgt + '/' + args.model
    save_path += '.pt'
    print(save_path)
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    best_metric = [0, 0, 0, 0]

    pretrain_tag = 10

    # === Train model
    unchange = 0
    tree = None
    for epoch in range(args.epochs):
        if (epoch + 1) % pretrain_tag == 0 :
            item_attr_embeddings = model.l2p(model.T.weight.data)
            tree = build_tree_gen(item_attr_embeddings, args.child_num, 4, model.sps)
        avg_loss = 0.
        avg_origin_loss = 0.
        avg_cluster_loss = 0.
        avg_cl_loss = 0.
        avg_pos_user = 0.
        avg_neg_user = 0.
        avg_pos_item = 0.
        avg_neg_item = 0.
        # === batch training
        t = time.time()
        for batch in range(num_batches):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data.adj_train_norm) 
            embeddings_noise = model.encode(data_drop.adj_train_norm)
            
            # embeddings = torch.nan_to_num(embeddings)
            
            if tree and use_user_cl_loss:
                train_loss, origin_loss, cluster_loss, item_cl_loss, pos_score_user, pos_score_item, neg_score_user, neg_score_item = model.compute_loss(embeddings, args.child_num, triples, tree, data, embeddings_noise, epoch)
            elif tree:
                train_loss, origin_loss, cluster_loss = \
                model_tgt.compute_loss(embeddings, args.child_num, triples, tree, data, embeddings_noise, embeddings, True, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
            
            else:
                train_loss = model_tgt.compute_loss(embeddings, args.child_num, triples, tree, data, embeddings_noise, embeddings, True, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
            
            loss = train_loss
            loss.backward()
            optimizer.step() 

            if tree and use_user_cl_loss:
                avg_loss += train_loss / num_batches
                avg_origin_loss += origin_loss / num_batches
                avg_cluster_loss += cluster_loss / num_batches
                avg_cl_loss += item_cl_loss / num_batches
                avg_pos_user += pos_score_user / num_batches
                avg_pos_item += pos_score_item / num_batches
                avg_neg_user += neg_score_user / num_batches
                avg_neg_item += neg_score_item / num_batches
            elif tree:
                avg_loss += train_loss / num_batches
                avg_origin_loss += origin_loss / num_batches
                avg_cluster_loss += cluster_loss / num_batches
            else:
                avg_loss += train_loss / num_batches
                     
                
        # === evaluate at the end of each batch
        if tree and use_user_cl_loss:
            avg_loss = avg_loss.detach().cpu().numpy()
            avg_origin_loss = avg_origin_loss.detach().cpu().numpy()
            avg_cluster_loss = avg_cluster_loss.detach().cpu().numpy()
            avg_cl_loss = avg_cl_loss.detach().cpu().numpy()
            avg_pos_user = avg_pos_user.detach().cpu().numpy()
            avg_pos_item = avg_pos_item.detach().cpu().numpy()
            avg_neg_user = avg_neg_user.detach().cpu().numpy()
            avg_neg_item = avg_neg_item.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                print(" ".join(['Epoch: {:04d}'.format(epoch),
                            'avg_loss: {:.3f}'.format(avg_loss),
                            'origin_loss: {:.3f}'.format(avg_origin_loss),
                            'cluster_loss: {:.3f}'.format(avg_cluster_loss),
                            'cl_loss: {:.3f}'.format(avg_cl_loss),
                            'avg_pos_user: {:.3f}'.format(avg_pos_user),
                            'avg_pos_item: {:.3f}'.format(avg_pos_item),
                            'avg_neg_user: {:.3f}'.format(avg_neg_user),
                            'avg_neg_item: {:.3f}'.format(avg_neg_item),
                            'time: {:.4f}s'.format(time.time() - t)]), end=' ')
                print("")
        
        elif tree:
            avg_loss = avg_loss.detach().cpu().numpy()
            avg_origin_loss = avg_origin_loss.detach().cpu().numpy()
            avg_cluster_loss = avg_cluster_loss.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                print(" ".join(['Epoch: {:04d}'.format(epoch),
                        'avg_loss: {:.3f}'.format(avg_loss),
                        'origin_loss: {:.3f}'.format(avg_origin_loss),
                        'cluster_loss: {:.3f}'.format(avg_cluster_loss),
                        'time: {:.4f}s'.format(time.time() - t)]), end=' ')
                print("")
        
        else:
            avg_loss = avg_loss.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                print(" ".join(['Epoch: {:04d}'.format(epoch),
                        'avg_loss: {:.3f}'.format(avg_loss),
                        'time: {:.4f}s'.format(time.time() - t)]), end=' ')
                print("")
            


        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start = time.time()
            embeddings = model.encode(data.adj_train_norm)
            pred_matrix = model.predict(embeddings, data)
            results = eval_rec(pred_matrix, data)
            recalls = results[0]
            if recalls[1] > best_metric[0]:
                best_metric[0] = recalls[1]
                best_metric[2] = results[1][1]
                torch.save(model.state_dict(), save_path)
                unchange = 0
            if recalls[2] > best_metric[1]:
                best_metric[1] = recalls[2]
                best_metric[3] = results[1][2]
                torch.save(model.state_dict(), save_path)
                unchange = 0
            else:
                unchange += args.eval_freq

            print('recall:'.join([str(round(x, 4)) for x in results[0]]))
            print('ndcg:'.join([str(round(x, 4)) for x in results[-1]]))
            print('best_eval', best_metric)
            if unchange == 200:
                print('best_eval', best_metric)
                break

    sampler.close()




def train2(model_tgt, model_src, dataset_tgt, dataset_src, data_tgt, data_src, sampler_tgt, sampler_src, data_drop_tgt, data_drop_src):
    # use_item_cl_loss = args.use_item_cl_loss
    use_user_cl_loss = args.use_user_cl_loss
    
    save_path_tgt = 'data/' + dataset_tgt + '/' + args.model + '.pt'
    save_path_src = 'data/' + dataset_src + '/' + args.model + '.pt'
    print(save_path_tgt)
    print(save_path_src)
    
    neighbor_neg = load_neighbor("Amazon-Movie", 3, 'a')
    # neighbor_neg = load_neighbor_dict("Amazon-Movie", 3)
    neighbor_tgt = load_neighbor_dict("Amazon-CD", 1)
    neighbor_src = load_neighbor_dict("Amazon-Movie", 1)
    
    
    optimizer_tgt = RiemannianSGD(params=model_tgt.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer_src = RiemannianSGD(params=model_src.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)

    num_pairs_tgt = data_tgt.adj_train.count_nonzero() // 2
    num_pairs_src = data_src.adj_train.count_nonzero() // 2
    
    num_batches = int(num_pairs_tgt / args.batch_size) + 1
    src_batch_size = int(num_pairs_src / num_batches)
    
    best_metric_tgt = [0, 0, 0, 0]
    best_metric_src = [0, 0, 0, 0]
    pretrain_tag = 10

    # === Train model
    unchange = 0
    tree = None
    tree_tgt=None
    tree_src=None
    for epoch in range(args.epochs):
        if (epoch + 1) % pretrain_tag == 0 :
            item_attr_embeddings_tgt = model_tgt.l2p(model_tgt.T.weight.data)
            tree_tgt = build_tree_gen(item_attr_embeddings_tgt, args.child_num, 4, model_tgt.sps)
            item_attr_embeddings_src = model_src.l2p(model_src.T.weight.data)
            tree_src = build_tree_gen(item_attr_embeddings_src, args.child_num, 4, model_src.sps)
        
        avg_loss_tgt = 0.
        avg_loss_src = 0.
        
        avg_origin_loss_tgt = 0.
        avg_origin_loss_src = 0.
        avg_cluster_loss_tgt = 0.
        avg_cluster_loss_src = 0.
        avg_trans_loss_tgt = 0.
        avg_pos_tgt = 0.
        avg_neg_tgt = 0.
        
        avg_cl_loss_tgt = 0.
        avg_pos_user_tgt = 0.
        avg_neg_user_tgt = 0.
        avg_pos_item_tgt = 0.
        avg_neg_item_tgt = 0.
        avg_cl_loss_src = 0.
        avg_pos_user_src = 0.
        avg_neg_user_src = 0.
        avg_pos_item_src = 0.
        avg_neg_item_src = 0.
        
        # === batch training
        t = time.time()
        for batch in range(num_batches):
            triples_tgt = sampler_tgt.next_batch()
            model_tgt.train()
            embeddings_tgt = model_tgt.encode(data_tgt.adj_train_norm)
            embeddings_noise_tgt = model_tgt.encode(data_drop_tgt.adj_train_norm)
            embeddings_tgt = torch.nan_to_num(embeddings_tgt)
            
            triples_src = sampler_src.next_batch()
            model_src.train()
            embeddings_src = model_src.encode(data_src.adj_train_norm) 
            embeddings_noise_src = model_src.encode(data_drop_src.adj_train_norm)
            embeddings_src = torch.nan_to_num(embeddings_src)
            
            
            if tree_tgt and tree_src and use_user_cl_loss:
                train_loss_tgt, origin_loss_tgt, cluster_loss_tgt, item_cl_loss_tgt, pos_score_user_tgt, pos_score_item_tgt, neg_score_user_tgt, neg_score_item_tgt, trans_loss_tgt = \
                model_tgt.compute_loss(embeddings_tgt, args.child_num, triples_tgt, tree_tgt, data_tgt, embeddings_noise_tgt, embeddings_src, True, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
                train_loss_src, origin_loss_src, cluster_loss_src, item_cl_loss_src, pos_score_user_src, pos_score_item_src, neg_score_user_src, neg_score_item_src = \
                model_src.compute_loss(embeddings_src, args.child_num, triples_src, tree_src, data_src, embeddings_noise_src, embeddings_tgt, False, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
            
            elif tree_tgt and tree_src:
                train_loss_tgt, origin_loss_tgt, cluster_loss_tgt, trans_loss_tgt, pos_tgt, neg_tgt = model_tgt.compute_loss(embeddings_tgt, args.child_num, triples_tgt, tree_tgt, data_tgt, embeddings_noise_tgt, embeddings_src, True, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
                train_loss_src, origin_loss_src, cluster_loss_src = model_src.compute_loss(embeddings_src, args.child_num, triples_src, tree_src, data_src, embeddings_noise_src, embeddings_tgt, False, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
            
            else:
                train_loss_tgt = model_tgt.compute_loss(embeddings_tgt, args.child_num, triples_tgt, tree_tgt, data_tgt, embeddings_noise_tgt, embeddings_src, True, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
                train_loss_src = model_src.compute_loss(embeddings_src, args.child_num, triples_src, tree_src, data_src, embeddings_noise_src, embeddings_tgt, False, neighbor_neg, neighbor_tgt, neighbor_src, epoch)
            
            optimizer_tgt.zero_grad()
            loss_tgt = train_loss_tgt
            loss_tgt.backward(retain_graph=True)
            optimizer_tgt.step()
            
            optimizer_src.zero_grad()
            loss_src = train_loss_src
            loss_src.backward()
            optimizer_src.step()

            if tree_tgt and tree_src and use_user_cl_loss:
                avg_loss_tgt += train_loss_tgt / num_batches
                avg_origin_loss_tgt += origin_loss_tgt / num_batches
                avg_cluster_loss_tgt += cluster_loss_tgt / num_batches
                avg_trans_loss_tgt += trans_loss_tgt / num_batches
               
                avg_cl_loss_tgt += item_cl_loss_tgt / num_batches
                avg_pos_user_tgt += pos_score_user_tgt / num_batches
                avg_pos_item_tgt += pos_score_item_tgt / num_batches
                avg_neg_user_tgt += neg_score_user_tgt / num_batches
                avg_neg_item_tgt += neg_score_item_tgt / num_batches
                
            
            elif tree_tgt and tree_src:
                avg_loss_tgt += train_loss_tgt / num_batches
                avg_origin_loss_tgt += origin_loss_tgt / num_batches
                avg_cluster_loss_tgt += cluster_loss_tgt / num_batches
                avg_trans_loss_tgt += trans_loss_tgt / num_batches
                avg_pos_tgt += pos_tgt / num_batches
                avg_neg_tgt += neg_tgt / num_batches
                
            else:
                avg_loss_tgt += train_loss_tgt / num_batches
                avg_loss_src += train_loss_src / num_batches
                
                
                
        # === evaluate at the end of each batch
        if tree_tgt and tree_src and use_user_cl_loss:
            avg_loss_tgt = avg_loss_tgt.detach().cpu().numpy()
            avg_origin_loss_tgt = avg_origin_loss_tgt.detach().cpu().numpy()
            avg_cluster_loss_tgt = avg_cluster_loss_tgt.detach().cpu().numpy()
            avg_cl_loss_tgt = avg_cl_loss_tgt.detach().cpu().numpy()
            avg_pos_user_tgt = avg_pos_user_tgt.detach().cpu().numpy()
            avg_pos_item_tgt = avg_pos_item_tgt.detach().cpu().numpy()
            avg_neg_user_tgt = avg_neg_user_tgt.detach().cpu().numpy()
            avg_neg_item_tgt = avg_neg_item_tgt.detach().cpu().numpy()
            avg_trans_loss_tgt = avg_trans_loss_tgt.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                print(" ".join(['Epoch: {:04d}'.format(epoch),
                            'avg_loss_tgt: {:.3f}'.format(avg_loss_tgt),
                            'origin_loss_tgt: {:.3f}'.format(avg_origin_loss_tgt),
                            'cluster_loss_tgt: {:.3f}'.format(avg_cluster_loss_tgt),
                            'cl_loss_tgt: {:.3f}'.format(avg_cl_loss_tgt),
                            'avg_pos_user_tgt: {:.3f}'.format(avg_pos_user_tgt),
                            'avg_pos_item_tgt: {:.3f}'.format(avg_pos_item_tgt),
                            'avg_neg_user_tgt: {:.3f}'.format(avg_neg_user_tgt),
                            'avg_neg_item_tgt: {:.3f}'.format(avg_neg_item_tgt),
                            'trans_loss_tgt: {:.3f}'.format(avg_trans_loss_tgt),
                            'time: {:.4f}s'.format(time.time() - t)]), end=' ')
                print("")
                
        
        elif tree_tgt and tree_src:
            avg_loss_tgt = avg_loss_tgt.detach().cpu().numpy()
            avg_origin_loss_tgt = avg_origin_loss_tgt.detach().cpu().numpy()
            avg_cluster_loss_tgt = avg_cluster_loss_tgt.detach().cpu().numpy()
            avg_trans_loss_tgt = avg_trans_loss_tgt.detach().cpu().numpy()
            avg_pos_tgt = avg_pos_tgt.detach().cpu().numpy()
            avg_neg_tgt = avg_neg_tgt.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                metrics = " ".join(['Epoch: {:04d}'.format(epoch),
                        'avg_loss_tgt: {:.3f}'.format(avg_loss_tgt),
                        'origin_loss_tgt: {:.3f}'.format(avg_origin_loss_tgt),
                        'cluster_loss_tgt: {:.3f}'.format(avg_cluster_loss_tgt),
                        'trans_loss_tgt: {:.3f}'.format(avg_trans_loss_tgt),
                        'pos_tgt: {:.3f}'.format(avg_pos_tgt),
                        'neg_tgt: {:.3f}'.format(avg_neg_tgt),
                        'time: {:.4f}s'.format(time.time() - t)])
                logger.info(metrics)
                print(metrics, end=' ')
                print("")
        
        else:
            avg_loss_tgt = avg_loss_tgt.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                metrics = " ".join(['Epoch: {:04d}'.format(epoch),
                        'avg_loss_tgt: {:.3f}'.format(avg_loss_tgt),
                        'time: {:.4f}s'.format(time.time() - t)])
                logger.info(metrics)
                print(metrics, end=' ')
                print("")
                
            avg_loss_src = avg_loss_src.detach().cpu().numpy()
            if (epoch + 1) % args.log_freq == 0:
                metrics = " ".join(['Epoch: {:04d}'.format(epoch),
                        'avg_loss_src: {:.3f}'.format(avg_loss_src),
                        'time: {:.4f}s'.format(time.time() - t)])
                logger.info(metrics)
                print(metrics, end=' ')
                print("")
            

        if (epoch + 1) % args.eval_freq == 0:
            model_tgt.eval()
            start = time.time()
            embeddings_tgt = model_tgt.encode(data_tgt.adj_train_norm)
            pred_matrix_tgt = model_tgt.predict(embeddings_tgt, data_tgt)
            results_tgt = eval_rec(pred_matrix_tgt, data_tgt)
            recalls_tgt = results_tgt[0]
            if recalls_tgt[1] > best_metric_tgt[0]:
                best_metric_tgt[0] = recalls_tgt[1]
                best_metric_tgt[2] = results_tgt[1][1]
                torch.save(model_tgt.state_dict(), save_path_tgt)
                unchange = 0
            if recalls_tgt[2] > best_metric_tgt[1]:
                best_metric_tgt[1] = recalls_tgt[2]
                best_metric_tgt[3] = results_tgt[1][2]
                torch.save(model_tgt.state_dict(), save_path_tgt)
                unchange = 0
            else:
                unchange += args.eval_freq

            print('recall_tgt:'.join([str(round(x, 4)) for x in results_tgt[0]]))
            print('ndcg_tgt:'.join([str(round(x, 4)) for x in results_tgt[-1]]))
            print('best_eval_tgt', best_metric_tgt)
            logger.info('recall_tgt:'.join([str(round(x, 4)) for x in results_tgt[0]]))
            logger.info('ndcg_tgt:'.join([str(round(x, 4)) for x in results_tgt[-1]]))
            logger.info('best_eval_tgt:'+ str(best_metric_tgt))
            if unchange == 200:
                print('best_eval_tgt', best_metric_tgt)
                break

    sampler_tgt.close()
    sampler_src.close()


def argmax_top_k(a, top_k=50):
    topk_score_items = []
    for i in range(len(a)):
        topk_score_item = heapq.nlargest(top_k, zip(a[i], itertools.count()))
        topk_score_items.append([x[1] for x in topk_score_item])
    return topk_score_items


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len - 1]
        dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def eval_rec(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = np.NINF
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20, 50]:
        recall.append(recall_at_k(data.test_dict, pred_list, k))

    all_ndcg = ndcg_func([*data.test_dict.values()], pred_list)
    ndcg = [all_ndcg[x - 1] for x in [5, 10, 20, 50]]

    return recall, ndcg


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='/home/featurize/work/Hyper_CDR/training.log', level=logging.INFO)
    
    args = parser.parse_args()
    print("args is:", args)
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    print(args.dim, args.lr, args.weight_decay, args.margin, args.batch_size)
    print(args.scale, args.num_layers, args.network, args.model)
    print(default_device())

    # === fix seed
    set_seed(args.seed)

    # === prepare data
    data_tgt = Data(args.dataset_tgt, args.norm_adj, args.seed, args.test_ratio, args.user_item_type, 0)
    data_drop_tgt = Data(args.dataset_tgt, args.norm_adj, args.seed, args.test_ratio, args.user_item_type, args.graph_dropout)
    total_edges_tgt = data_tgt.adj_train.count_nonzero()
    args.n_nodes_tgt = data_tgt.num_users + data_tgt.num_items
    args.feat_dim = args.embedding_dim

    # === negative sampler (iterator)
    sampler_tgt = WarpSampler((data_tgt.num_users, data_tgt.num_items), data_tgt.adj_train, args.batch_size, args.num_neg)
    model_tgt = TaxoRec((data_tgt.num_users, data_tgt.num_items), args, args.dataset_tgt)
    model_tgt = model_tgt.to(default_device())
    
    for name, param in model_tgt.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model_tgt.parameters()).device)


    if args.trans_loss:
        data_src = Data(args.dataset_src, args.norm_adj, args.seed, args.test_ratio, args.user_item_type, 0)  
        data_drop_src = Data(args.dataset_src, args.norm_adj, args.seed, args.test_ratio, args.user_item_type, args.graph_dropout)
        total_edges_src = data_src.adj_train.count_nonzero()
        args.n_nodes_src = data_src.num_users + data_src.num_items
        sampler_src = WarpSampler((data_src.num_users, data_src.num_items), data_src.adj_train, args.batch_size, args.num_neg)
        model_src = TaxoRec((data_src.num_users, data_src.num_items), args, args.dataset_src)
        model_src = model_src.to(default_device())
        for name, param in model_src.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        print('model is running on', next(model_src.parameters()).device)

    try:
        if args.trans_loss:
            train2(model_tgt, model_src, args.dataset_tgt, args.dataset_src, data_tgt, data_src, sampler_tgt, sampler_src, data_drop_tgt, data_drop_src)
        else:
            train(model_tgt, data_tgt, sampler_tgt, data_drop_tgt)
    
    except Exception:
        if args.trans_loss:
            sampler_src.close()
        
        sampler_tgt.close()
        traceback.print_exc()