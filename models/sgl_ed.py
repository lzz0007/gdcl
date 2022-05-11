# coding: utf-8
# """
# Paper: Self-supervised Graph Learning for Recommendation
# Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian & Xing Xie
# Reference: https://github.com/hexiangnan/LightGCN
# """
#
# Updated by Zhou Xin (enoche.chow@gmail.com)
# Time: 2021/07/30


import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import BPRLoss, L2Loss


class SGL_ED(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    def __init__(self, config, dataset):
        super(SGL_ED, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.ssl_reg = config['ssl_reg']
        self.ssl_ratio = config['ssl_ratio']
        self.ssl_temp = config['ssl_temp']

        self.bpr_loss = BPRLoss()
        self.l2_loss = L2Loss()

        self.embedding_dict = self._init_model()

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.sub_mat = {}

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        return embedding_dict

    def pre_epoch_processing(self):
        # build up random adj matrix
        self.sub_mat['sub_mat_1'] = self.create_random_sub_adj_mat().to(self.device)
        self.sub_mat['sub_mat_2'] = self.create_random_sub_adj_mat().to(self.device)

    def create_random_sub_adj_mat(self):
        # build up random adj matrix
        inter_M = self.interaction_matrix
        training_user = inter_M.row # no of data in the matrix
        training_item = inter_M.col
        training_len = len(training_user)
        n_nodes = self.n_users + self.n_items
        keep_idx = np.random.choice(np.arange(training_len),
                                    size=int(training_len * (1 - self.ssl_ratio)), replace=False)
        user_np = np.array(training_user)[keep_idx]
        item_np = np.array(training_item)[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1)) + 1e-7        # avoid RuntimeWarning: divide by zero encountered in power
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        coo = adj_matrix.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        data = torch.FloatTensor(coo.data)
        sparse_adj = torch.sparse.FloatTensor(i, data, torch.Size(coo.shape))

        return sparse_adj

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        # user_embeddings = self.user_embedding.weight
        # item_embeddings = self.item_embedding.weight
        # ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]
        for layer_idx in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj_matrix, ego_embeddings)
            all_embeddings.append(ego_embeddings)

            ego_embeddings_sub1 = torch.sparse.mm(self.sub_mat['sub_mat_1'], ego_embeddings_sub1)
            all_embeddings_sub1.append(ego_embeddings_sub1)

            ego_embeddings_sub2 = torch.sparse.mm(self.sub_mat['sub_mat_2'], ego_embeddings_sub2)
            all_embeddings_sub2.append(ego_embeddings_sub2)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

        all_embeddings_sub1 = torch.stack(all_embeddings_sub1, dim=1)
        all_embeddings_sub1 = torch.mean(all_embeddings_sub1, dim=1)
        u_g_embeddings_sub1, i_g_embeddings_sub1 = torch.split(all_embeddings_sub1, [self.n_users, self.n_items])

        all_embeddings_sub2 = torch.stack(all_embeddings_sub2, dim=1)
        all_embeddings_sub2 = torch.mean(all_embeddings_sub2, dim=1)
        u_g_embeddings_sub2, i_g_embeddings_sub2 = torch.split(all_embeddings_sub2, [self.n_users, self.n_items])

        return u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2
        # return u_g_embeddings, i_g_embeddings

    def ssl_loss_v2(self, a_embeddings_sub1, a_embeddings_sub2, elements):
        user_emb1 = a_embeddings_sub1[elements]
        user_emb2 = a_embeddings_sub2[elements]

        normalize_user_emb1 = F.normalize(user_emb1, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, dim=1)
        normalize_all_user_emb2 = F.normalize(a_embeddings_sub2, 1)
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2), dim=1)
        ttl_score_user = torch.matmul(normalize_user_emb1, normalize_all_user_emb2.t())

        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)

        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))
        return ssl_loss_user

    def create_bpr_loss(self, ua_embeddings, ia_embeddings, users, pos_items, neg_items):
        batch_u_embeddings = ua_embeddings[users]
        batch_pos_i_embeddings = ia_embeddings[pos_items]
        batch_neg_i_embeddings = ia_embeddings[neg_items]
        raw_u_embeddings = self.embedding_dict['user_emb']
        raw_i_embeddings = self.embedding_dict['item_emb']
        batch_u_embeddings_pre =  raw_u_embeddings[users]
        batch_pos_i_embeddings_pre = raw_i_embeddings[pos_items]
        batch_neg_i_embeddings_pre = raw_i_embeddings[neg_items]
        regularizer = self.l2_loss(batch_u_embeddings_pre, batch_pos_i_embeddings_pre, batch_neg_i_embeddings_pre)
        emb_loss = self.reg_weight * regularizer

        pos_scores = torch.mul(batch_u_embeddings, batch_pos_i_embeddings).sum(dim=1)
        neg_scores = torch.mul(batch_u_embeddings, batch_neg_i_embeddings).sum(dim=1)

        #ori_bpr_loss = self.bpr_loss(pos_scores, neg_scores)
        m = torch.nn.LogSigmoid()
        bpr_loss = torch.sum(-m(pos_scores - neg_scores))

        return bpr_loss, emb_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, ua_embeddings_sub1, ia_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub2 = self.forward()

        # both_side loss
        ssl_loss_user = self.ssl_loss_v2(ua_embeddings_sub1, ua_embeddings_sub2, users)
        ssl_loss_item = self.ssl_loss_v2(ia_embeddings_sub1, ia_embeddings_sub2, pos_items)
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)

        bpr_loss, emb_loss = self.create_bpr_loss(ua_embeddings, ia_embeddings, users, pos_items, neg_items)

        loss = bpr_loss + emb_loss + ssl_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e, _, _, _, _ = self.forward()
        # restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        return scores

