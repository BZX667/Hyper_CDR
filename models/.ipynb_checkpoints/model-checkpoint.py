import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz
import scipy.sparse as sp
from utils.helper import default_device
import math
import torch.nn.functional as F
from geoopt import Lorentz
import geoopt.manifolds.lorentz.math as lorentz_math
from geoopt import PoincareBall
from geoopt import ManifoldParameter
import models.encoders as encoders
import time
import random
import json
import pandas as pd
import torch.nn.functional as F
# # 设置随机种子
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)
eps = 1e-15


class TaxoRec(nn.Module):

    def __init__(self, users_items, args, dataset):
        super(TaxoRec, self).__init__()

        self.c = torch.tensor([args.c]).to(default_device())

        self.manifold = Lorentz(args.c)
        self.ball = PoincareBall(args.c)
        self.encoder = getattr(encoders, "HG")(args.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.args = args

        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(default_device())

        self.embedding.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight']), True)
        self.embedding.weight = ManifoldParameter(self.embedding.weight, self.manifold, True)

        tag_matrix = load_npz('data/' + dataset + '/item_tag_matrix.npz')
        tag_labels = tag_matrix.A
        tmp = np.sum(tag_labels, axis=1, keepdims=True)
        tag_labels = tag_labels / (tmp+1)
        # tag_labels = tag_labels / tmp
        
        
        self.num_tags = tag_labels.shape[1]
        self.T = nn.Embedding(num_embeddings=self.num_tags,
                              embedding_dim=args.dim).to(default_device())
        self.T.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.T.weight = nn.Parameter(self.manifold.expmap0(self.T.state_dict()['weight']), requires_grad=True)
        self.T.weight = ManifoldParameter(self.T.weight, self.manifold, requires_grad=True)

        self.ugr = nn.Embedding(num_embeddings=self.num_users,
                                embedding_dim=args.dim).to(default_device())
        self.ugr.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.ugr.weight = nn.Parameter(self.manifold.expmap0(self.ugr.state_dict()['weight']))
        self.ugr.weight = ManifoldParameter(self.ugr.weight, self.manifold, requires_grad=True)

        self.sps = torch.from_numpy(tag_labels).float().to(default_device())

        self.lam = args.lam
        
        self.use_user_cl_loss = args.use_user_cl_loss
        self.cluster_loss_weight = args.cluster_loss_weight
        self.cl_loss_weight = args.cl_loss_weight
        self.trans_loss = args.trans_loss
        self.trans_epoch = args.trans_epoch

    def encode(self, adj):
        adj = adj.to(default_device())

        x1 = self.manifold.projx(self.embedding.weight)
        h_in = self.encoder.encode(x1, adj)
        
        # print('a_shape')
        # print(adj.shape)
        # print('x_shape')
        # print(x1.shape)
        # print('h_in shape')
        # print(h_in.shape)

        emb_tag = self.manifold.projx(self.T.weight)
        emb_tag_in = self.manifold.projx(self.ugr.weight)
        emb_tag_weight = self.sps
        emb_tag_out = self.hyper_agg(emb_tag_weight, emb_tag)
        x2 = torch.cat([emb_tag_in, emb_tag_out], dim=0)
        h_gr = self.encoder.encode(x2, adj)
        h = torch.cat([h_in, h_gr], dim=-1)
        return h

    def lorentz_factor(self, x, dim=-1, keepdim=False):
        """
        Parameters
        ----------
        x : tensor
            point on Klein disk
        c : float
            negative curvature
        dim : int
            dimension to calculate Lorenz factor
        keepdim : bool
            retain the last dim? (default: false)
        Returns
        -------
        tensor
            Lorenz factor
        """
        return 1 / torch.sqrt((1 - self.c.to(x.device) * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(eps))

    def hyper_agg(self, weights, x):
        x_p = self.l2p(x)
        x_k = self.p2k(x_p)
        gamma = self.lorentz_factor(x_k.detach(), dim=-1, keepdim=True)
        mean = weights.matmul(gamma * x_k) / weights.matmul(gamma).clamp_min(eps)
        # return self.p2l(mean)

        return self.p2l(self.k2p(mean))

    def hyper_agg_uniform(self, x):
        x_p = self.l2p(x)
        x_k = self.p2k(x_p)
        gamma = self.lorentz_factor(x_k.detach(), dim=-1, keepdim=True)

        mean = (gamma * x_k) / (gamma.clamp_min(eps))

        return self.p2l(self.k2p(mean))

    def p2l(self, x):
        return lorentz_math.poincare_to_lorentz(x, k=self.c)

    def l2p(self, x):
        return lorentz_math.lorentz_to_poincare(x, k=self.c)

    def p2k(self, x):
        denom = 1 + self.c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x):
        denom = 1 + torch.sqrt(1 - self.c * x.pow(2).sum(-1, keepdim=True).clamp_min(eps))
        return x / denom

    def add_gaussian_noise(self, embeddings, std_dev):
        noise = torch.randn_like(embeddings) * std_dev
        # noise = torch.dropout(noise, 0.5, train=True)
        noisy_embeddings = embeddings + noise
        return noisy_embeddings
    
  
    def decode(self, h_all, idx):
        # print(h_all.size())
        # h_all[torch.isnan(h_all)] = 0
        # print(torch.isnan(h_all).sum().item())
        h = h_all[:, :self.args.embedding_dim]
        emb_in = h[idx[:, 0]]
        emb_out = h[idx[:, 1]]
        
        # print(torch.isnan(emb_in).sum().item())
        # print(torch.isnan(emb_out).sum().item())
        
        sqdist = self.manifold.dist2(emb_in, emb_out, keepdim=True).clamp_max(50.0)
        assert not torch.isnan(sqdist).any()
        assert not torch.isinf(self.T.weight).any()
        assert not torch.isnan(self.T.weight).any()
        # print(self.T.weight)
        h2 = h_all[:, self.args.embedding_dim:]
        emb_tag_in = h2[idx[:, 0]]
        emb_tag_out = h2[idx[:, 1]]
        
        assert not torch.isnan(emb_tag_out).any()
        assert not torch.isinf(emb_tag_out).any()
        assert not torch.isnan(emb_tag_in).any()
        sqdist += self.manifold.dist2(emb_tag_in, emb_tag_out, keepdim=True).clamp_max(15.0)
        return sqdist
    
    def decode2(self, emb1, emb2):
        emb_in = emb1[:, :self.args.embedding_dim]
        emb_out = emb2[:, :self.args.embedding_dim]
        sqdist = self.manifold.dist2(emb_in, emb_out, keepdim=True).clamp_max(50.0)
        assert not torch.isnan(sqdist).any()
        assert not torch.isinf(self.T.weight).any()
        assert not torch.isnan(self.T.weight).any()
        
        emb_tag_in = emb1[:, self.args.embedding_dim:]
        emb_tag_out = emb2[:, self.args.embedding_dim:]
        assert not torch.isnan(emb_tag_out).any()
        assert not torch.isinf(emb_tag_out).any()
        assert not torch.isnan(emb_tag_in).any()
        sqdist += self.manifold.dist2(emb_tag_in, emb_tag_out, keepdim=True).clamp_max(15.0)
        assert not torch.isnan(sqdist).any()
        return sqdist

    def shuffle_vector(self, node_emb):
        indices = torch.randperm(node_emb.size(0))
        new_node_emb = node_emb.clone()
        new_node_emb = new_node_emb[indices]
        return new_node_emb
    
    # def shuffle_vector(self, node_emb):
    #     new_node_emb = node_emb.clone()
    #     random.shuffle(new_node_emb)
    #     return new_node_emb

    def node_cl_loss(self, h, data, node_type, temperature, noise_coef, embeddings_noise):
        num_users, num_items = data.num_users, data.num_items
        if node_type == 'user':
            node_emb = h[:num_users, :]
            node_emb_noise = embeddings_noise[:num_users, :]
        elif node_type == 'item':
            node_emb = h[np.arange(num_users, num_users + num_items), :]
            node_emb_noise = embeddings_noise[np.arange(num_users, num_users + num_items), :]

        # node_emb_noise = self.add_gaussian_noise(node_emb, noise_coef)
        pos_scores = self.decode2(node_emb, node_emb_noise)
        # pos_scores = self.decode2(node_emb, node_emb_graphdropout)
        assert not torch.isnan(pos_scores).any()
        neg_emb = self.shuffle_vector(node_emb)
        neg_scores = self.decode2(node_emb, neg_emb)
        
        # BPR loss
        dis = pos_scores-neg_scores + 0.5
        dis = dis/temperature
        loss = torch.relu(dis)
        loss = torch.sum(loss)
        # # dis = neg_scores - pos_scores
        # # print("pos_score is:", pos_scores)
        # # print("neg_score is:", neg_scores)
        # # print("temperature is:", temperature)
        # assert not torch.isnan(dis).any()
        # loss = torch.sigmoid(dis / temperature)
        # # print("loss is:", loss[loss<0])
        # # print("loss is:", loss)
        # loss = torch.log(loss)
        # # assert not torch.isnan(loss).any()
        # # print("[after log]loss is:", loss)
        # loss = -torch.sum(loss)
        # # print("[after mean] loss is:", loss)
        # weight = 0.001
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        return loss, torch.sum(pos_scores), torch.sum(neg_scores)
    
    
    def cluster_loss(self, tree, child_num=5):
        loss = 0
        for k in tree.keys():
            # each height
            if k == 0:
                continue
            node_list = tree[k]
            for i in range(len(node_list)):
                node = node_list[i].term_ids
                if len(node) == 0 or len(node) == 1:
                    continue
                try:
                    scores = node_list[i].scores.cuda(default_device())
                except Exception as e:
                    print(node_list[i].term_ids, k, i)
                scores = scores / scores.max()
                node_terms = self.T(torch.LongTensor(node).cuda(default_device()))

                center = self.hyper_agg(scores, node_terms).repeat(len(node)).view(len(node), -1)

                assert not torch.isnan(center).any()
                loss += ((node_terms - center) ** 2).sum()
        return loss

    def find_first_order_neighbors(self, adj_csr, user_index):
        neighbor_indices = adj_csr[user_index].indices
        return neighbor_indices[:-1]
    
    def find_higher_order_neighbors(self, adj_csr, user_index, order):    
        current_neighbors = set(adj_csr[user_index].indices)    
        higher_order_neighbors = set()    
        visited = set(current_neighbors)    
        for _ in range(order - 1):        
            next_neighbors = set()        
            for neighbor in current_neighbors:            
                next_neighbors.update(adj_csr[neighbor].indices)        
            next_neighbors -= visited        
            higher_order_neighbors.update(next_neighbors)        
            visited.update(next_neighbors)        
            current_neighbors = next_neighbors
        return list(higher_order_neighbors)
    

    def find_first_and_second_order_neighbors(self, adj_csr, overlap_size, dataset):
        first_order_neighbors_list = []
        second_order_neighbors_list = []
        for user_index in overlap_size:
            first_order_neighbors = adj_csr[user_index].indices
            second_order_neighbors = set()
            for neighbor in first_order_neighbors:
                second_order_neighbors.update(adj_csr[neighbor].indices)
            second_order_neighbors -= set(first_order_neighbors)  # 去除一阶邻居
            first_order_neighbors_list.append(list(first_order_neighbors))
            second_order_neighbors_list.append(list(second_order_neighbors))
        df = pd.DataFrame({'1st': first_order_neighbors_list, '2nd': second_order_neighbors_list})
        df.to_csv('/home/featurize/work/Hyper_CDR/data/' + dataset + '/neighbor.csv', index=False)

    
    def create_neighbor_dict(self, overlap_size, adj, order, dataset):
        neighbor_indices_dict = {}
        if order == 1:
            for i in range(overlap_size):
                neighbor_indices = self.find_first_order_neighbors(adj, i)
                neighbor_indices = str(neighbor_indices).strip('[]')
                neighbor_indices_dict[i] = neighbor_indices
        elif order > 1:
            for i in range(overlap_size):
                neighbor_indices = self.find_higher_order_neighbors(adj, i, order)
                neighbor_indices = str(neighbor_indices).strip('[]')
                neighbor_indices_dict[i] = neighbor_indices
        
        with open('/home/featurize/work/Hyper_CDR/data/' + dataset + '/neighbor_{}_dict.json'.format(order), 'w') as file:
            json.dump(neighbor_indices_dict, file)
            
        
    def compute_subgraph_emb(self, overlap_size, neighbor_dict, embedding, order, dataset):
        embedding_matrix = embedding.clone()
        for user_index in range(overlap_size):
            neighbor_indices = neighbor_dict[str(user_index)]
            if order==1:
                neighbor_indices = [int(x) for x in neighbor_indices.split(" ")]
            else:
                neighbor_indices = [int(x) for x in neighbor_indices.split(", ")[:10]]
            neighbor_embeddings = embedding[neighbor_indices]   
            neighbor_embeddings = neighbor_embeddings.unsqueeze(1)
            # mean_embeddings = F.avg_pool1d(neighbor_embeddings, kernel_size=62)
            mean_embeddings = torch.mean(neighbor_embeddings, dim=0)   
            embedding_matrix[user_index] = mean_embeddings
        return embedding_matrix[:overlap_size]  
    
    
    def generate_neg_emb(self, overlap_size, neighbor_dict, embedding, order, dataset):
        embedding_clone = embedding.clone()
        for user_index in range(overlap_size):
            neighbor_indices = neighbor_dict[str(user_index)]
            if order==1:
                neighbor_indices = [int(x) for x in neighbor_indices.split(" ")]
            else:
                neighbor_indices = [int(x) for x in neighbor_indices.split(", ")[:10]]
            neighbor_embeddings = embedding[neighbor_indices]    
            mean_embeddings = torch.mean(neighbor_embeddings, dim=0)
            embedding_matrix[user_index] = mean_embeddings
        return embedding_clone[:overlap_size, :]
    
    
    def generate_neg_emb2(self, overlap_size, neighbor, embedding):
        embedding_clone = embedding.clone()
        neighbor_list = list(neighbor['3nd_neighbor'])
        embedding_clone[:overlap_size] = embedding[neighbor_list]  
        return embedding_clone[:overlap_size, :]    
    
    
    def compute_dis(self, origin_emb, pos_emb, neg_emb, temperature):
        pos_scores = self.decode2(origin_emb, pos_emb)
        assert not torch.isnan(pos_scores).any()

        neg_scores = self.decode2(origin_emb, neg_emb) + 0.1
        dis = (pos_scores - neg_scores)
        dis = dis / temperature
        dis = torch.relu(dis)
        dis = torch.sum(dis)
        assert not torch.isnan(dis).any()
        assert not torch.isinf(dis).any()
        return dis, torch.sum(pos_scores), torch.sum(neg_scores)

    def trans_domain_loss(self, emb_tgt, overlap_size, temperature, emb_src, neighbor_neg, neighbor_tgt, neighbor_src):
        # loss 1 
        # pos_dis: user_a in target domain <<< user_a in source domain
        # neg_dis: user_a in target domain >>> user_b in source domain
        overlap_emb_tgt = emb_tgt[:overlap_size, :]
        overlap_emb_src = emb_src[:overlap_size, :]
        
        # self.create_neighbor_dict(overlap_size, adj_src, 3, "Amazon-CD")
        # self.find_first_and_second_order_neighbors(adj_tgt, overlap_size, "Amazon-CD")
        # assert 1==2
        
        # neg_sample = emb_src[overlap_size:2*overlap_size, :]
        # neg_sample = self.shuffle_vector(neg_sample)
        neg_sample = self.generate_neg_emb2(overlap_size, neighbor_neg, emb_src)
        # neg_sample = self.compute_subgraph_emb(overlap_size, neighbor_neg, emb_src, 3, "Amazon-Movie")       
        trans_loss,pos,neg = self.compute_dis(overlap_emb_tgt, overlap_emb_src, neg_sample, temperature)
        
#         # loss 2
#         # pos_dis: user_a 1st neighbor mean value <<< user_a in source domain
#         # neg_dis: user_a 1st neighbor mean value >>> user_b in soource domain 
        # overlap_emb_tgt_subgraph = self.compute_subgraph_emb(overlap_size, neighbor_tgt, emb_tgt, 1, "Amazon-CD")
#         overlap_emb_src_subgraph= self.compute_subgraph_emb(overlap_size, neighbor_src, emb_src, 1, "Amazon-Movie")
#         # neg_sample = self.compute_subgraph_emb(overlap_size, neighbor, emb_src, 3, "Amazon-Movie")
#         neg_sample = self.generate_neg_emb2(overlap_size, neighbor_neg, emb_tgt)
        # loss2 = self.compute_dis(overlap_emb_src, overlap_emb_tgt_subgraph, neg_sample, temperature)

        return 0.1*(trans_loss).clamp_max(5000.0),pos,neg
        
        
    def compute_loss(self, embeddings, child_num, triples, tree, data, embeddings_noise, embeddings_trans, activate_trans, neighbor_neg, neighbor_tgt, neighbor_src, epoch):
        assert not torch.isnan(triples).any()
        triples = triples.to(default_device())
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in
                           sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)
        origin_loss = pos_scores - neg_scores + self.margin
        origin_loss[origin_loss < 0] = 0
        origin_loss = torch.sum(origin_loss)
        loss = origin_loss.clone()
        
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        
        if tree and self.use_user_cl_loss and self.lam>0:
            cluster_loss = self.lam * self.cluster_loss(tree, child_num)
            loss += self.cluster_loss_weight * cluster_loss
            # print('user.....')
            cl_loss1, pos_scores_1, neg_scores_1 = self.node_cl_loss(embeddings, data, 'user', self.args.user_temperature, 0.01, embeddings_noise)
            # print("item....")
            cl_loss2, pos_scores_2, neg_scores_2 = self.node_cl_loss(embeddings, data, 'item', self.args.item_temperature, 0.01, embeddings_noise)
            cl_loss = cl_loss1 + cl_loss2
            cl_loss = self.cl_loss_weight * cl_loss
            loss += cl_loss
            
            if self.trans_loss:
                if activate_trans:
                    trans_loss, pos, neg = self.trans_domain_loss(embeddings, self.args.overlap_size, self.args.user_temperature, embeddings_trans, neighbor_neg, neighbor_tgt, neighbor_src)
                    if epoch < self.trans_epoch:
                        loss += trans_loss
                    return loss, origin_loss, cluster_loss, cl_loss, self.cl_loss_weight * pos_scores_1, self.cl_loss_weight * pos_scores_2, self.cl_loss_weight * neg_scores_1, self.cl_loss_weight * neg_scores_2, trans_loss

                return loss, origin_loss, cluster_loss, cl_loss, self.cl_loss_weight * pos_scores_1, self.cl_loss_weight * pos_scores_2, self.cl_loss_weight * neg_scores_1, self.cl_loss_weight * neg_scores_2

        
        elif tree and self.lam>0:
            cluster_loss = self.lam * self.cluster_loss(tree, child_num)
            loss += self.cluster_loss_weight * cluster_loss
            if self.trans_loss:
                if activate_trans:
                    trans_loss,pos,neg = self.trans_domain_loss(embeddings, self.args.overlap_size, self.args.user_temperature, embeddings_trans, neighbor_neg, neighbor_tgt, neighbor_src)
                    if epoch < self.trans_epoch:
                        loss += trans_loss
                    return loss, origin_loss, cluster_loss, trans_loss, pos, neg
            return loss, origin_loss, cluster_loss
        
        else:
            return loss


    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :self.args.embedding_dim]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :self.args.embedding_dim]
            sqdist = self.manifold.dist2(emb_in, emb_out)

            emb_tag_in = h[:, self.args.embedding_dim:][i].repeat(num_items).view(num_items, -1)
            emb_tag_out = h[np.arange(num_users, num_users + num_items), self.args.embedding_dim:]
            sqdist += self.manifold.dist2(emb_tag_in, emb_tag_out)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix