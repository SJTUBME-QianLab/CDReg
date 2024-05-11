import pandas as pd
import numpy as np
import os
import argparse
from sklearn import metrics

data_path = './../data/simulation/'
save_path = './../results/'


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--method', default='', type=str, help='model name')
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
    else:
        raise ValueError('method')

    final_weight, metric_i = performance(sco, fea_info, individuals)
    df_groupby1, df_groupby2, df_true_std = group_check(fea_info, final_weight)

    pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'eval_FS.xlsx'))
    final_weight.to_excel(pd_writer, index=False, index_label=True, sheet_name="final_weight")
    metric_i.to_excel(pd_writer, index=False, index_label=True, sheet_name="metrices")
    df_groupby1.to_excel(pd_writer, index=True, index_label=True, sheet_name="group_check1")
    df_groupby2.to_excel(pd_writer, index=True, index_label=True, sheet_name="group_check2")
    df_true_std.to_excel(pd_writer, index=True, index_label=True, sheet_name="df_true_std")
    pd_writer.save()


def get_data(data_dir, data_name):
    if 'cov' in data_name:
        # test_idx = np.loadtxt(os.path.join(data_dir, data_name, 'test_idx', '%d.txt' % fold)).astype(int)
        # train_idx = np.array(list(set(np.arange(len(Yall))) - set(test_idx)), dtype=int)
        data = None
        fea_info = pd.read_csv(os.path.join(data_dir, data_name, 'basic_info.csv')).iloc[:, 1:]
        # fea_name, gp_label, loc, true_01, isol_01, beta
        individuals = np.load(os.path.join(data_dir, data_name, 'spac_idx.npy'))
    elif data_name == 'LUAD':
        Xall = np.load(os.path.join(data_dir, 'X_normL2.npy'))
        Yall = np.load(os.path.join(data_dir, 'Y.npy'))
        assert len(Xall) == (492 + 183)
        train_idx = list(range(492))
        test_idx = list(range(492, 492+183))
        data = [Xall[train_idx, :], Yall[train_idx], Xall[test_idx, :], Yall[test_idx]]
        fea_info = pd.read_csv(os.path.join(data_dir, 'info.csv')).iloc[1:, :]
        # IlmnID,gene_set,gene_num,UCSC_RefGene_Name,CHR,MAPINFO,gp_size,gp_idx
        individuals = None
    elif data_name == 'AD':
        # test_idx = np.loadtxt(os.path.join(data_dir, 'test_idx', '%d.txt' % fold)).astype(int)
        # train_idx = np.array(list(set(np.arange(len(Yall))) - set(test_idx)), dtype=int)
        data = None
        fea_info = pd.read_csv(os.path.join(data_dir, 'info.csv')).iloc[1:, :]
        # IlmnID,gene_set,gene_num,UCSC_RefGene_Name,CHR,MAPINFO,gp_size,gp_idx
        individuals = None
    else:
        raise ValueError(data_name)

    return data, fea_info, individuals


def performance(sco, fea_info, individuals):
    fea_name = fea_info['fea_name'].values
    fea_idxes = fea_info.index.values
    locs = fea_info['loc'].values
    gp_info = fea_info['gp_label'].values
    true_beta = fea_info['beta'].values
    TorF = fea_info['true_01'].values
    indi01 = np.zeros_like(TorF)
    indi01[individuals] = 1

    sco.columns = ['index', 'final_weight']
    sco.insert(loc=2, column='abs_weight', value=abs(sco['final_weight']))
    sco_rank = sco.sort_values(['abs_weight'], ascending=False, kind='mergesort').reset_index(drop=True)

    # slc_idx = np.hstack([sco_rank['index'][:top_num], sco_rank['index'][int(np.sum(TorF))]])

    # basic info
    info = pd.DataFrame({
        'NO': range(1, len(fea_name) + 1),
        'fea_ind': fea_idxes,
        'fea_name': fea_name,
        'ch': gp_info,
        'loc': locs,
    })
    info = pd.merge(info, sco, left_on='fea_ind', right_on='index').drop(['index'], axis=1, inplace=False)

    # Feature selection performance
    true_beta_df = pd.DataFrame({'true_beta': true_beta, 'indi01': indi01})
    # TorF = (abs(beta) > 0).astype(float)
    true_beta_df.insert(1, 'TorF', TorF)
    final_weight = pd.concat([info, true_beta_df], axis=1)
    final_weight.insert(loc=list(final_weight.columns).index('abs_weight') + 1,
                        column='abs_weight_normalize',
                        value=final_weight[['abs_weight']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),
                                                                 axis=0))

    pred_top = np.zeros((1, len(TorF)))
    slc_idx = sco_rank['index'][:int(np.sum(TorF))]
    pred_top[0, slc_idx] = 1
    prob_1 = final_weight['abs_weight_normalize']
    prob = np.vstack([1 - prob_1, prob_1]).T
    metric_i, fpr, tpr = evaluate(prob, true=final_weight['TorF'].values)

    TPR = np.sum((pred_top * TorF)) / np.sum(TorF)
    sparsity = np.sum(final_weight['abs_weight'] > 0) / len(final_weight)

    indi_weight = final_weight.loc[individuals, 'abs_weight_normalize'].mean()

    metric_i = pd.DataFrame(np.array([[TPR, sparsity, indi_weight] + metric_i]),
                            columns=['top-TPR', 'sparsity', 'indi_abs_w_mean',
                                     'acc', 'roc_auc', 'recall', 'precision', 'f1score', 'specificity'])

    return final_weight, metric_i


def group_check(fea_info, final_weight):
    df = fea_info.reset_index(drop=False, inplace=False)
    df.rename(columns={df.columns[0]: 'index'}, inplace=True)
    df[['index']] = (df[['index']] + 1).astype(int)
    df.loc[np.where(df[['isol_01']] == 1)[0], 'true_01'] = 2
    df.drop(['index', 'fea_name', 'loc', 'beta'], axis=1, inplace=True)

    df = pd.concat([df, final_weight[['abs_weight', 'abs_weight_normalize']]], axis=1)

    df_groupby1 = df.groupby(['true_01', 'gp_label']).mean()
    df_groupby2 = df.groupby(['gp_label', 'true_01']).mean()
    df_true_std = df[df['true_01'] == 1].groupby(['gp_label']).std()

    return df_groupby1, df_groupby2, df_true_std


def evaluate(prob, true_onehot=None, true=None):
    # calculate
    pred = np.argmax(prob, axis=1)
    if true_onehot is None and true is None:
        raise ValueError
    if true_onehot is None:
        true_onehot = pd.get_dummies(true).values
    if true is None:
        true = np.argmax(true_onehot, axis=1)
    acc = metrics.accuracy_score(true, pred)
    recall = metrics.recall_score(true, pred, average='macro')
    precision = metrics.precision_score(true, pred, average='macro')
    f1score = metrics.f1_score(true, pred, average='macro')

    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    # print([tn, fp, fn, tp])
    fpr, tpr, _ = metrics.roc_curve(true_onehot.ravel(), prob.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    specificity = tn / (tn + fp)

    return [acc, roc_auc, recall, precision, f1score, specificity], fpr, tpr


if __name__ == '__main__':
    main()
