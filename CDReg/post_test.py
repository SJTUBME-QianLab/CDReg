import os
import numpy as np
import pandas as pd
import argparse
import random
from joblib import Parallel, delayed
from itertools import product, permutations

import sys
sys.path.append('./../comparing')
from eval_R_app import get_score
from train_test import eval_6clf

root_dir = './../results/'
data_root = './../data/'
save0_dir = './../results/testing/'
method_list0 = ['LASSO', 'Enet0.8', 'SGLASSO', 'dmp10', 'DMRcate']


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--test_type', default='resample', type=str)
    parser.add_argument('--fold_num', default=30, type=int)
    args = parser.parse_args()

    data_name = args.data_name
    seed = args.seed
    test_type = args.test_type
    fold_num = args.fold_num
    top_num = 50

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if data_name in ['AD', 'LUAD']:
        data_dir = os.path.join(data_root, data_name, 'group_gene')
        fea_info = pd.read_csv(os.path.join(data_dir, 'info.csv')).iloc[1:, :]
        # IlmnID,gene_set,gene_num,UCSC_RefGene_Name,CHR,MAPINFO,gp_size,gp_idx
        head = 'IlmnID'
        if data_name == 'LUAD':
            method_list = method_list0 + ['L10.5_L210.20_Lg1.2_Lcon0.3', 'L10.5_L210.20_Lg0.0_Lcon0.3', 'L10.5_L210.20_Lg1.2_Lcon0.0']
            folds_list = [88, 99, 115, 121, 150, 195, 204, 229, 246, 263]
        elif data_name == 'AD':
            method_list = method_list0 + ['pclogit', 'L10.150_L210.040_Lg0.5_Lcon1.2_lr1.00']
            folds_list = [13, 39, 214, 313, 337, 346, 398, 400, 443, 444]
    elif data_name == 'CHR20p0.05gp20':
        data_dir = os.path.join(data_root, data_name)
        fea_info = pd.read_csv(os.path.join(data_dir, 'info.csv')).iloc[1:, :]
        head = 'probe'
        method_list = method_list0 + ['pclogit', 'L10.5L210.1Ls1.2Lc1_lr0.0001']
        folds_list = [14, 21, 24, 71, 73, 78, 79, 106, 110, 125]
    else:
        raise ValueError(f'wrong data_name: {data_name}')

    def sub_fold(kk):
        data = get_data_test(data_dir, data_name,
                                  seed=seed, fold_num=fold_num, fold=kk, test_type=test_type)
        for method in method_list:
            print(method, kk)
            new_save_dir = os.path.join(save0_dir, 'results_' + data_name, method,
                                        's'+str(seed)+'_'+test_type, 'fold'+str(kk))
            os.makedirs(new_save_dir, exist_ok=True)
            if os.path.isfile(os.path.join(new_save_dir, 'acc_allT.csv')):
                continue

            print(method, kk)
            if method in ['dmp10', 'DMRcate']:
                result_dir = os.path.join(root_dir, data_name, 'other', method)
            elif method in ['LASSO', 'Enet0.8', 'SGLASSO', 'pclogit']:
                result_dir = os.path.join(root_dir, data_name, '3m_default100_0.0001', method)
            elif method.startswith('L1') and data_name in ['LUAD', 'AD']:
                result_dir = os.path.join(root_dir, data_name, method)
            elif method.startswith('L1') and data_name.startswith('CHR'):
                result_dir = os.path.join(root_dir, data_name, method, 's1')
            else:
                raise ValueError(f'wrong method: {method}')
            sco = get_score(method, data_name, fea_info, head, result_dir)

            eval_clf_new(data, fea_info, sco, result_dir=new_save_dir, top_num=top_num,
                         descending=False if method in ['DMRcate'] else True)

    Parallel(n_jobs=2)(delayed(sub_fold)(fold) for fold in folds_list)


def get_data_test(data_dir, data_name, seed, fold_num=None, fold=None, test_type=None):
    if data_name == 'LUAD':
        Xall = np.load(os.path.join(data_dir, 'X_normL2.npy'))
        Yall = np.load(os.path.join(data_dir, 'Y.npy'))
        assert len(Xall) == (492 + 183)
        train_idx = list(range(492))
        test_idx = list(range(492, 492 + 183))

        if fold is None and test_type == 'onetime':
            data = [Xall[train_idx, :], Yall[train_idx], Xall[test_idx, :], Yall[test_idx]]

        elif fold is not None and test_type == 'resample':
            resample_num = int(183 * 0.5)  # 91
            random.seed(seed)
            test_idx_list = []
            save_idx_dir = os.path.join(data_dir, 'resample_{}_test_idx'.format(resample_num))
            os.makedirs(save_idx_dir, exist_ok=True)
            for i in range(fold_num):
                test = random.sample(test_idx, resample_num)
                test_idx_list.append(test)
                if os.path.isfile(os.path.join(save_idx_dir, '%d.txt' % i)):
                    test_save = np.loadtxt(os.path.join(save_idx_dir, '%d.txt' % i)).astype(int)
                    assert (test_save == test).all()
                else:
                    np.savetxt(os.path.join(save_idx_dir, '%d.txt' % i), test, fmt="%d")
            data = [Xall[train_idx, :], Yall[train_idx], Xall[test_idx_list[fold], :], Yall[test_idx_list[fold]]]

        else:
            raise ValueError(test_type)

    elif data_name in ['AD'] or data_name.startswith('CHR'):
        Xall = np.load(os.path.join(data_dir, 'X_normL2.npy'))
        Yall = np.load(os.path.join(data_dir, 'Y.npy'))
        if fold is not None and test_type == 'resample':
            resample_num = int(len(Yall) * 0.2)  # AD: 1001*0.2=200; PC: 353*0.2=70
            random.seed(seed)
            all_idx = list(range(len(Yall)))
            test_idx_list, train_idx_list = [], []
            save_idx_dir = os.path.join(data_dir, 'resample_{}_test_idx'.format(resample_num))
            os.makedirs(save_idx_dir, exist_ok=True)
            for i in range(fold_num):
                test = random.sample(all_idx, resample_num)
                train = list(set(all_idx) - set(test))
                test_idx_list.append(test)
                train_idx_list.append(train)
                if os.path.isfile(os.path.join(save_idx_dir, '%d.txt' % i)):
                    test_save = np.loadtxt(os.path.join(save_idx_dir, '%d.txt' % i)).astype(int)
                    assert (test_save == test).all()
                else:
                    np.savetxt(os.path.join(save_idx_dir, '%d.txt' % i), test, fmt="%d")
            data = [Xall[train_idx_list[fold], :], Yall[train_idx_list[fold]],
                    Xall[test_idx_list[fold], :], Yall[test_idx_list[fold]]]
        else:
            raise ValueError(test_type)

    else:
        raise ValueError(data_name)

    return data


def eval_clf_new(data, fea_info, sco, result_dir, top_num=30, descending=True):
    if 'IlmnID' in fea_info.columns:
        fea_name = fea_info['IlmnID'].values.tolist()
        locs = fea_info['MAPINFO'].values.astype(float).astype(int)
    elif 'probe' in fea_info.columns:
        fea_name = fea_info['probe'].values.tolist()
        locs = fea_info['start'].values.astype(int)
    else:
        raise ValueError(fea_info.columns[:4])
    gp_info = fea_info['gp_idx'].values.astype(int)
    fea_idxes = np.arange(len(fea_name))

    sco.columns = ['index', 'final_weight']
    sco.insert(loc=2, column='abs_weight', value=abs(sco['final_weight']))
    sco_rank = sco.sort_values(['abs_weight'], ascending=not descending, kind='mergesort').reset_index(drop=True)
    slc_idx = sco_rank['index'][:top_num]

    df_all_eval, info = eval_6clf(slc_idx, fea_name, fea_idxes, locs, gp_info, data)

    acc_TF = []
    for i, df in df_all_eval.items():
        if i == 'svm':
            acc_TF.append(df.iloc[:, :list(df.columns).index('acc') + 1])
        else:
            acc_TF.append(df[['acc']])
    acc_TF = pd.concat(acc_TF, axis=1)
    acc_TF.columns = list(acc_TF.columns[:-(len(df_all_eval))]) + list(df_all_eval.keys())

    pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'final_%d_eval0.xlsx' % top_num))
    for i, df in df_all_eval.items():
        df.to_excel(pd_writer, index=False, index_label=True, sheet_name=i)
    pd_writer.save()

    acc_TF.to_csv(os.path.join(result_dir, 'acc_allT.csv'), index=False)


if __name__ == '__main__':
    main()
