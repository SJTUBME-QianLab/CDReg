import pandas as pd
import numpy as np
import os
import argparse

import sys
sys.path.append('./../CDReg')
from train_test import get_data, performance_app

root_dir = './../results/'
data_root = './../data/'
method_list0 = ['LASSO', 'Enet0.8', 'SGLASSO', 'dmp10', 'DMRcate']


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--seed', type=int, default=2020)  # random seed
    # parser.add_argument('--fold', default=0, type=int, help='fold idx for cross-validation')
    parser.add_argument('--set_name', default='3m_default100_0.0001', type=str, help='set name')
    parser.add_argument('--choice', default=1, type=int, help='choose evaluation strategy')
    args = parser.parse_args()
    set_seed(args.seed)
    eval_3m(args.data_name, args.set_name, args.choice)


def eval_3m(data_name, set_name, choice):
    if data_name in ['AD', 'LUAD']:
        head = 'IlmnID'
        if data_name == 'AD':
            # method_list = ['DMRcate']
            method_list = method_list0 + ['pclogit', 'L10.150_L210.040_Lg0.5_Lcon1.2_lr1.00']
        elif data_name == 'LUAD':
            # method_list = ['DMRcate']
            method_list = method_list0 + ['L10.5_L210.20_Lg1.2_Lcon0.3', 'L10.5_L210.20_Lg1.2_Lcon0.0', 'L10.5_L210.20_Lg0.0_Lcon0.3']
    elif data_name == 'CHR20p0.05gp20':
        head = 'probe'
        method_list = method_list0 + ['pclogit', 'L10.5L210.1Ls1.2Lc1_lr0.0001']
    else:
        raise ValueError(f'wrong data_name: {data_name}')
    top_num = 50

    data, fea_info, individuals = get_data(data_root, data_name)

    sparcity_all = []
    for method in method_list:
        print(method)
        if method in ['dmp', 'DMRcate']:
            result_dir = os.path.join(root_dir, data_name, 'other', method)
        elif method in ['LASSO', 'Enet0.8', 'SGLASSO', 'pclogit']:
            result_dir = os.path.join(root_dir, data_name, set_name, method)
        elif method.startswith('L1'):
            if data_name.startswith('CHR'):
                result_dir = os.path.join(root_dir, data_name, method, 's1')
            else:
                result_dir = os.path.join(root_dir, data_name, method)
        else:
            raise ValueError(f'wrong method: {method}')

        sco = get_score(method, data_name, fea_info, head, result_dir)
        df_all_eval, final_weight, sparsity = performance_app(sco, data, fea_info, top_num=top_num,
                                                              descending=False if method in ['dmp', 'DMRcate'] else True,
                                                              choice=choice)
        sparcity_all.append(sparsity)

        if df_all_eval is not None:
            acc_TF = []
            for i, df in df_all_eval.items():
                if i == 'svm':
                    acc_TF.append(df.iloc[:, :list(df.columns).index('acc') + 1])
                else:
                    acc_TF.append(df[['acc']])
            acc_TF = pd.concat(acc_TF, axis=1)
            acc_TF.columns = list(acc_TF.columns[:-(len(df_all_eval))]) + list(df_all_eval.keys())

        pd_writer = pd.ExcelWriter(os.path.join(result_dir, f'final_{top_num}_eval{choice}.xlsx'))
        for i, df in df_all_eval.items():
            df.to_excel(pd_writer, index=False, index_label=True, sheet_name=i)
        pd_writer.save()

        pd_writer = pd.ExcelWriter(os.path.join(result_dir, f'eval_FS{choice}.xlsx'))
        if df_all_eval is not None:
            acc_TF.to_excel(pd_writer, index=False, index_label=True, sheet_name="acc_allT")
        final_weight.to_excel(pd_writer, index=False, index_label=True, sheet_name="final_weight")
        pd.DataFrame([sparsity]).to_excel(pd_writer, index=False, index_label=True, sheet_name="sparsity")
        pd_writer.save()


def get_score(method, data_name, fea_info, head, result_dir):
    if method in ['LASSO', 'Enet0.8']:
        sco = pd.read_csv(os.path.join(result_dir, '%s_coef_lam10_re1.csv' % method))
        sco = sco.iloc[1:, :]
        sco.iloc[:, 0] = [int(x.split('V')[1]) - 1 for x in sco.iloc[:, 0]]
    elif method == 'SGLASSO':
        sco = pd.read_csv(os.path.join(result_dir, '%s_coef_lam10_re1.csv' % method))
        sco.iloc[:, 0] = [int(x) - 1 for x in sco.iloc[:, 0]]
    elif method == 'pclogit':
        sco = pd.read_csv(os.path.join(result_dir, '%s_coef_lam10_re1.csv' % method))
        sco.iloc[:, 0] = [int(x.split('V')[1]) - 1 for x in sco.iloc[:, 0]]
    elif method == 'MHB':
        sco = pd.read_csv(os.path.join(result_dir, 'fea_select.csv'))
        sco = sco.iloc[:, [0, sco.shape[1] - 1]]
    elif method == 'dmp10':
        sco = pd.read_csv(os.path.join(result_dir, 'coef.csv'))
        sco.columns = [head] + sco.columns[1:].tolist()
        sco = pd.merge(fea_info[[head]], sco, on=head, how='left')
        sco.reset_index(drop=False, inplace=True)
        sco = sco.loc[:, ['index', 'beta']]
    elif method == 'DMRcate':
        sco = pd.read_csv(os.path.join(result_dir, 'coef.csv'))
        sco = sco.iloc[:, 1:]
        assert all(sco.sort_values('Fisher', ascending=True, kind='stable').index == sco.index)
        fea_info['DMR'] = ''
        sco.insert(loc=0, column='DMR',
                   value=[f'{x}:{y}-{z}' for x, y, z in zip(sco['seqnames'], sco['start'], sco['end'])])
        if data_name in ['CHRYp0.05gp20']:
            assert 'chrY' in sco['seqnames'].unique()
            sco = sco.sort_values(['start'], ascending=True).reset_index(drop=True)
            for dmr in sco.index:
                chro, start, end = sco.loc[dmr, ['seqnames', 'start', 'end']]
                fea_info.loc[(fea_info['#chr'] == chro) & (fea_info['start'] >= start) & (
                            fea_info['start'] <= end), 'DMR'] = f'{chro}:{start}-{end}'
        elif data_name.startswith('CHR'):
            assert 'chrY' not in sco['seqnames'].unique()
            sco.insert(loc=1, column='chr', value=[int(x.split('chr')[1]) for x in sco['seqnames']])
            sco = sco.sort_values(['chr', 'start'], ascending=True).reset_index(drop=True)
            for dmr in sco.index:
                chro, start, end = sco.loc[dmr, ['seqnames', 'start', 'end']]
                fea_info.loc[(fea_info['#chr'] == chro) & (fea_info['start'] >= start) & (
                            fea_info['start'] <= end), 'DMR'] = f'{chro}:{start}-{end}'
        else:
            assert data_name in ['AD', 'LUAD']
            sco.insert(loc=1, column='chr', value=[int(x.split('chr')[1]) for x in sco['seqnames']])
            sco = sco.sort_values(['chr', 'start'], ascending=True).reset_index(drop=True)
            for dmr in sco.index:
                chro, start, end = sco.loc[dmr, ['seqnames', 'start', 'end']]
                fea_info.loc[(fea_info['CHR'] == int(chro.split('chr')[1])) & (fea_info['MAPINFO'] >= start) & (
                            fea_info['MAPINFO'] <= end), 'DMR'] = f'{chro}:{start}-{end}'
        sco = pd.merge(fea_info, sco, on='DMR', how='left')
        if os.path.isfile(os.path.join(root_dir, data_name, 'other', method, 'DMR_info.csv')):
            sco_old = pd.read_csv(os.path.join(root_dir, data_name, 'other', method, 'DMR_info.csv'))
            sco_old['DMR'].fillna('', inplace=True)
            pd.testing.assert_frame_equal(sco_old, sco, check_dtype=False)
        else:
            sco.to_csv(os.path.join(root_dir, data_name, 'other', method, 'DMR_info.csv'), index=False)
        sco.reset_index(drop=False, inplace=True)
        sco = sco.loc[:, ['index', 'Fisher']]
    elif method.startswith('L1'):
        sco = pd.read_csv(os.path.join(result_dir, 'fea_scores.csv'))
        sco = sco.iloc[:, [0, sco.shape[1] - 1]]
    else:
        raise ValueError('method')
    return sco


def set_seed(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    main()
