# coding: utf-8
# @Time   : 2021/3/30
# @Author : Xin Zhou
# @Email  : enoche.chow@gmail.com

# UPDATE:

"""
Main entry
##########################
"""


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
from utils.quick_start import quick_start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str, default='GDCL', help='name of models') # LGCN_Diff_Hete
    parser.add_argument('--dataset', '-d', type=str, default='arts', help='name of datasets') # ml-100k
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', default=999, type=int, help='seed number')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train_batch_size', '-bs', type=int, default=2048, help='batch size')
    parser.add_argument('--reg_weight', '-wd', nargs='+', type=float, default=[1e-05], help='model regularization weight')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'seed': args.seed,
        'learning_rate': args.learning_rate,
        'train_batch_size': args.train_batch_size,
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

