# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN

BUIR implementation
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import BPRLoss, EmbLoss, SSLLoss
from models.common.init import xavier_uniform_initialization

import time
from data import ppr


class GDCL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GDCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.embedding_dict = self._init_model()

        # generate normalized adj matrix
        self.adj = self.get_adj_mat()
        self.norm_adj_matrix = self.sparse_mat_to_tensor(self.get_norm_adj_mat(self.adj)).to(self.device)

        # generate ppr matrix
        self.ppr_mat = self.get_ppr(config['dataset'], config['alpha_u'], config['alpha_i'], config['eps'],
                                    config['topu'], config['topi'], config['ppr_norm'])

        self.diff_users_mat = self.ppr_mat[:self.n_users, :self.n_users].tocoo()
        self.diff_items_mat = self.ppr_mat[self.n_users:, self.n_users:].tocoo()
        self.diff_ui_mat = self.get_diff_ui_mat(self.ppr_mat, config['dataset'], config['alpha_u'], config['alpha_i'],
                                            config['eps'], config['topu'], config['topi'], config['ppr_norm'])

        self.diff_users = self.sparse_mat_to_tensor(self.diff_users_mat.tocoo()).to(self.device)
        self.diff_items = self.sparse_mat_to_tensor(self.diff_items_mat.tocoo()).to(self.device)
        self.diff_ui = self.sparse_mat_to_tensor(self.diff_ui_mat.tocoo()).to(self.device)

        self.ssl_loss = SSLLoss(config['ssl_temp'])
        self.ssl_reg = config['ssl_reg']
        self.ssl_ratio = config['ssl_ratio']

        self.prob = config['prob']
        self.pred = config['pred']
        self.full = config['full']

    def create_random_sub_adj_mat(self, mat):
        mat = mat.tocoo()
        training_rows = mat.row
        training_cols = mat.col
        training_data = mat.data
        training_len = len(training_rows)
        n_nodes = mat.shape[0]

        if self.prob == 'True':
            keep_idx = np.random.choice(np.arange(training_len),
                                        size=int(training_len * (1 - self.ssl_ratio)), replace=False,
                                        p=training_data/sum(training_data))
        else:
            keep_idx = np.random.choice(np.arange(training_len),
                                        size=int(training_len * (1 - self.ssl_ratio)), replace=False)
        rows_np = np.array(training_rows)[keep_idx]
        cols_np = np.array(training_cols)[keep_idx]
        data_np = np.array(training_data)[keep_idx]
        res = sp.csr_matrix((data_np, (rows_np, cols_np)), shape=(n_nodes, n_nodes))
        return res

    def pre_epoch_processing(self):
        if self.ssl_ratio == 0:
            pass
        else:
            if self.full == 'True':
                # sampling for entire diff
                ppr_mat = self.create_random_sub_adj_mat(self.ppr_mat)
                diff_users_mat = ppr_mat[:self.n_users, :self.n_users].tocoo()
                diff_items_mat = ppr_mat[self.n_users:, self.n_users:].tocoo()

                n_nodes = ppr_mat.shape[0]
                diff_ui_mat_1 = ppr_mat[:self.n_users, self.n_users:].tocoo()
                diff_ui_mat_2 = ppr_mat[self.n_users:, :self.n_users].tocoo()
                diff_ui_mat_rows = np.concatenate((diff_ui_mat_1.row, diff_ui_mat_2.row + self.n_users), axis=None)
                diff_ui_mat_cols = np.concatenate((diff_ui_mat_1.col + self.n_users, diff_ui_mat_2.col), axis=None)
                diff_ui_mat_data = np.concatenate((diff_ui_mat_1.data, diff_ui_mat_2.data), axis=None)
                diff_ui_mat = sp.csr_matrix((diff_ui_mat_data, (diff_ui_mat_rows, diff_ui_mat_cols)), shape=(n_nodes, n_nodes))
            else:
                # sampling for each block
                diff_users_mat = self.create_random_sub_adj_mat(self.diff_users_mat)
                diff_items_mat = self.create_random_sub_adj_mat(self.diff_items_mat)
                diff_ui_mat = self.create_random_sub_adj_mat(self.diff_ui_mat)

            # split diffusion mat
            self.diff_users = self.sparse_mat_to_tensor(diff_users_mat.tocoo()).to(self.device)
            self.diff_items = self.sparse_mat_to_tensor(diff_items_mat.tocoo()).to(self.device)
            self.diff_ui = self.sparse_mat_to_tensor(diff_ui_mat.tocoo()).to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        return embedding_dict

    def get_adj_mat(self):
        """
        Get the interaction matrix of users and items.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        return A

    def get_norm_adj_mat(self, A):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return L

    def get_ppr(self, ds, alpha_u, alpha_i, eps, topu, topi, ppr_norm):
        fname = f'{ds}_s_alpha_{str(alpha_u)}_{str(alpha_i)}_eps_{str(eps)}_topu_{str(topu)}_topi_{str(topi)}_{str(ppr_norm)}_sep.npz'
        try:
            topk_mat = sp.load_npz('data/' + fname)
        except:
            train_idx = np.array([i for i in range(int(self.n_users + self.n_items))])
            topk_mat = ppr.topk_ppr_matrix(self.n_users, self.n_items, self.adj.tocsr(), alpha_u, alpha_i, eps,
                                           train_idx, topu, topi,
                                           normalization=ppr_norm)
            sp.save_npz('data/' + fname, topk_mat)
        return topk_mat

    def get_diff_ui_mat(self, mat, ds, alpha_u, alpha_i, eps, topu, topi, ppr_norm):
        fname = f'{ds}_s_alpha_{str(alpha_u)}_{str(alpha_i)}_eps_{str(eps)}_topu_{str(topu)}_topi_{str(topi)}_{str(ppr_norm)}_sep_diff_ui.npz'
        try:
            mat = sp.load_npz('data/' + fname)
        except:
            n_nodes = mat.shape[0]
            diff_ui_mat_1 = mat[:self.n_users, self.n_users:].tocoo()
            diff_ui_mat_2 = mat[self.n_users:, :self.n_users].tocoo()
            diff_ui_mat_rows = np.concatenate((diff_ui_mat_1.row, diff_ui_mat_2.row + self.n_users), axis=None)
            diff_ui_mat_cols = np.concatenate((diff_ui_mat_1.col + self.n_users, diff_ui_mat_2.col), axis=None)
            diff_ui_mat_data = np.concatenate((diff_ui_mat_1.data, diff_ui_mat_2.data), axis=None)
            diff_ui_mat = sp.csr_matrix((diff_ui_mat_data, (diff_ui_mat_rows, diff_ui_mat_cols)), shape=(n_nodes, n_nodes))

            sp.save_npz('data/' + fname, diff_ui_mat.tocsr())
        return mat.tocoo()

    def sparse_mat_to_tensor(self, L):
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
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        all_embeddings_ppr = self.get_ego_embeddings()
        embs, embs_ppr = [], []

        # adj GCN
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embs.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embs, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings = lightgcn_all_embeddings[:self.n_users, :]
        item_all_embeddings = lightgcn_all_embeddings[self.n_users:, :]

        # ppr GCN
        user_emb_ppr = all_embeddings_ppr[:self.n_users, :]
        user_emb_ppr = torch.sparse.mm(self.diff_users, user_emb_ppr)

        item_emb_ppr = all_embeddings_ppr[self.n_users:, :]
        item_emb_ppr = torch.sparse.mm(self.diff_items, item_emb_ppr)

        all_emb_ppr = torch.sparse.mm(self.diff_ui, all_embeddings_ppr)

        user_ppr_all_embeddings = all_emb_ppr[:self.n_users, :]
        item_ppr_all_embeddings = all_emb_ppr[self.n_users:, :]

        user_ppr_all_embeddings = torch.mean(torch.stack([user_emb_ppr, user_ppr_all_embeddings], dim=1), dim=1)
        item_ppr_all_embeddings = torch.mean(torch.stack([item_emb_ppr, item_ppr_all_embeddings], dim=1), dim=1)

        return user_all_embeddings, user_ppr_all_embeddings, item_all_embeddings, item_ppr_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        # sum to combine ppr and adj
        user_all_embeddings, user_ppr_all_embeddings, item_all_embeddings, item_ppr_all_embeddings = self.forward()
        user_embeddings = user_all_embeddings+user_ppr_all_embeddings
        item_embeddings = item_all_embeddings+item_ppr_all_embeddings

        # embeddings of interaction pairs
        u_embeddings = user_embeddings[user, :]
        posi_embeddings = item_embeddings[pos_item, :]
        negi_embeddings = item_embeddings[neg_item, :]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR reg Loss
        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)

        # calculate ssl loss
        ssl_loss = self.ssl_loss(user_all_embeddings[user, :], user_ppr_all_embeddings[user, :],
                                 item_all_embeddings[pos_item, :], item_ppr_all_embeddings[pos_item, :])
        return mf_loss, self.reg_weight * reg_loss, self.ssl_reg * ssl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        if self.pred == 'True':
            # use non-sampling diffusion mat
            self.diff_users = self.sparse_mat_to_tensor(self.diff_users_mat.tocoo()).to(self.device)
            self.diff_items = self.sparse_mat_to_tensor(self.diff_items_mat.tocoo()).to(self.device)
            self.diff_ui = self.sparse_mat_to_tensor(self.diff_ui_mat.tocoo()).to(self.device)

            restore_user_e, restore_user_e_ppr, restore_item_e, restore_item_e_ppr = self.forward()

            # use adj and diff for prediction
            restore_user_e = restore_user_e + restore_user_e_ppr
            restore_item_e = restore_item_e + restore_item_e_ppr
        else:
            restore_user_e, restore_user_e_ppr, restore_item_e, restore_item_e_ppr = self.forward()

        u_embeddings = restore_user_e[user, :]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
