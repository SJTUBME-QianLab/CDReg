import pandas as pd
import numpy as np
import os
import argparse

import sys
sys.path.append('./../CDReg')
from train_test import get_data, performance, group_check

data_path = './../data/simulation/'
save_path = './../results/'


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--method', default='', type=str, help='model name')
    parser.add_argument('--choice', default=None, type=int, help='choose evaluation strategy')
    # parser.add_argument('--fold', default=0, type=int, help='fold idx for cross-validation')
    args = parser.parse_args()

    data, fea_info, individuals = get_data(data_path, args.data_name)
    result_dir = os.path.join(save_path, args.data_name, args.method, f's{args.seed}')
    if args.method in ['LASSO', 'Enet0.8']:
        sco = pd.read_csv(os.path.join(result_dir, 'coef_lam10_re1.csv'))
        sco = sco.iloc[1:, :]
        sco.iloc[:, 0] = [int(x.split('V')[1]) - 1 for x in sco.iloc[:, 0]]
    elif args.method == 'SGLASSO':
        sco = pd.read_csv(os.path.join(result_dir, 'coef_lam10_re1.csv'))
        sco.iloc[:, 0] = [int(x) - 1 for x in sco.iloc[:, 0]]
    elif args.method == 'pclogit':
        sco = pd.read_csv(os.path.join(result_dir, 'coef_lam10_re1.csv'))
        sco.iloc[:, 0] = [int(x.split('V')[1]) - 1 for x in sco.iloc[:, 0]]
    elif args.method == 'MHB':
        sco = pd.read_csv(os.path.join(result_dir, 'fea_select.csv'))
        sco = sco.iloc[:, [0, sco.shape[1]-1]]
    elif args.method == 'dmp10':
        sco = pd.read_csv(os.path.join(result_dir, 'coef.csv'))
        sco.iloc[:, 0] = [int(x.split('V')[1]) - 1 for x in sco.iloc[:, 0]]
        sco.columns = ['index'] + sco.columns[1:].tolist()
        sco = sco.loc[:, ['index', 'beta']]
        fea_num = len(fea_info)
        sco_ = pd.DataFrame({
            'index': range(fea_num)
        })
        sco = pd.merge(sco_, sco, on='index', how='left')
        sco.replace(np.nan, 0, inplace=True)
    else:
        raise ValueError('method')

    final_weight, metric_i = performance(sco, fea_info, individuals,
                                         descending=False if args.method in ['dmp'] else True, choice=args.choice)
    df_groupby1, df_groupby2, df_true_std = group_check(fea_info, final_weight)

    pd_writer = pd.ExcelWriter(os.path.join(result_dir, f'eval_FS{args.choice}.xlsx'))
    final_weight.to_excel(pd_writer, index=False, index_label=True, sheet_name="final_weight")
    metric_i.to_excel(pd_writer, index=False, index_label=True, sheet_name="metrices")
    df_groupby1.to_excel(pd_writer, index=True, index_label=True, sheet_name="group_check1")
    df_groupby2.to_excel(pd_writer, index=True, index_label=True, sheet_name="group_check2")
    df_true_std.to_excel(pd_writer, index=True, index_label=True, sheet_name="df_true_std")
    pd_writer.save()


if __name__ == '__main__':
    main()
