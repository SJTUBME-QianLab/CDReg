import pandas as pd
import numpy as np
import os
import argparse
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
root_dir = './../results/'
data_root = './../data/'


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--seed', type=int, default=2020)  # random seed
    # parser.add_argument('--fold', default=0, type=int, help='fold idx for cross-validation')
    parser.add_argument('--set_name', default='', type=str, help='set name')
    args = parser.parse_args()
    set_seed(args.seed)
    eval_3m(args.data_name, args.set_name)


def eval_3m(data_name, set_name):
    method_list = ['LASSO', 'Enet0.8', 'SGLASSO', 'pclogit', 'MHB']
    data_dir = os.path.join(data_root, data_name, 'group_gene')
    R_dir = os.path.join(root_dir, data_name, set_name)

    top_num = 50

    data, fea_info, individuals = get_data(data_dir, data_name)

    sparcity_all = []
    for method in method_list:
        result_dir = os.path.join(R_dir, method)
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
            sco = sco.iloc[:, [0, sco.shape[1]-1]]
        else:
            raise ValueError('method')

        df_all_eval, final_weight, sparsity = performance_app(sco, data, fea_info, top_num=top_num)
        sparcity_all.append(sparsity)

        if df_all_eval is not None:
            pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'final_%d_eval0.xlsx' % top_num))
            acc_TF = []
            for i, df in df_all_eval.items():
                df.to_excel(pd_writer, index=False, index_label=True, sheet_name=i)
                if i == 'svm':
                    acc_TF.append(df.iloc[:, :list(df.columns).index('acc')+1])
                else:
                    acc_TF.append(df[['acc']])
            pd_writer.save()
            acc_TF = pd.concat(acc_TF, axis=1)
            acc_TF.columns = list(acc_TF.columns[:-(len(df_all_eval))]) + list(df_all_eval.keys())

        pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'eval_FS.xlsx'))
        if df_all_eval is not None:
            acc_TF.to_excel(pd_writer, index=False, index_label=True, sheet_name="acc_allT")
        final_weight.to_excel(pd_writer, index=False, index_label=True, sheet_name="final_weight")
        pd.DataFrame([sparsity]).to_excel(pd_writer, index=False, index_label=True, sheet_name="sparsity")
        pd_writer.save()

    # concat classification
    # main_func = ConcatParams(R_dir, method_list)
    # metrices = main_func.sparsity()
    # acc_svm = main_func.acc_svm('svm')
    # acc_nb = main_func.acc_svm('nb')
    # acc_lr = main_func.acc_svm('lr')
    # acc_rf = main_func.acc_svm('rf')
    #
    # pd_writer = pd.ExcelWriter(os.path.join(R_dir, 'eval_R.xlsx'))
    # metrices.to_excel(pd_writer, index=True, index_label=True, sheet_name="sparsity")
    # acc_svm.to_excel(pd_writer, index=True, index_label=True, sheet_name="acc_svm")
    # acc_nb.to_excel(pd_writer, index=True, index_label=True, sheet_name="acc_nb")
    # acc_lr.to_excel(pd_writer, index=True, index_label=True, sheet_name="acc_lr")
    # acc_rf.to_excel(pd_writer, index=True, index_label=True, sheet_name="acc_rf")
    # pd_writer.save()


# class ConcatParams:
#     def __init__(self, root_dir, method_list):
#         self.root_dir = root_dir
#         self.method_list = method_list
#
#     def acc_svm(self, clf):
#         evals = []
#         for method in self.method_list:
#             eval_FS = pd.read_excel(os.path.join(self.root_dir, method, 'eval_FS.xlsx'), sheet_name='acc_allT')
#             evals.append(eval_FS[[clf]])
#         evals = pd.concat(evals, axis=1)
#         evals.columns = self.method_list
#         return evals
#
#     def sparsity(self):
#         evals = []
#         for method in self.method_list:
#             eval_FS = pd.read_excel(os.path.join(self.root_dir, method, 'eval_FS.xlsx'), sheet_name='sparsity')
#             evals.append(eval_FS.iloc[0, 0])
#         evals = pd.DataFrame([evals], columns=self.method_list)
#         return evals


class GetClf:
    def __init__(self, root_dir, fold, method, R_flag=None):
        if R_flag:
            self.root_dir = os.path.join(root_dir, R_flag, 'fold%d' % fold, method)
        else:
            self.root_dir = os.path.join(root_dir, method, 'fold%d' % fold)
        self.method = method
        if os.path.isfile(os.path.join(self.root_dir, 'final_50_eval0.xlsx')):
            self.eval = True
        else:
            self.eval = False
        self.clf = pd.read_excel(os.path.join(self.root_dir, 'final_50_eval0.xlsx'), sheet_name=None)

    def metric(self, clf, metric):
        df = self.clf[clf][[metric]]
        return df.values

    def fea_slc(self):
        df = pd.read_excel(os.path.join(self.root_dir, 'eval_FS.xlsx'), sheet_name='final_weight')[['abs_weight_normalize']]
        return df.values


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


def performance_app(sco, data, fea_info, top_num=50):
    fea_name = fea_info['IlmnID'].values.tolist()
    fea_idxes = np.arange(len(fea_name))
    locs = fea_info['MAPINFO'].values.astype(float).astype(int)
    gp_info = fea_info['gp_idx'].values.astype(int)

    sco.columns = ['index', 'final_weight']
#     sco['index'] = sco['index'] + 1
    sco.insert(loc=2, column='abs_weight', value=abs(sco['final_weight']))
    sco_rank = sco.sort_values(['abs_weight'], ascending=False, kind='mergesort').reset_index(drop=True)

#     slc_idx = np.hstack([sco_rank['index'][:top_num], sco_rank['index'][top_num]])
    slc_idx = sco_rank['index'][:top_num]

    if data is None:
        info = pd.DataFrame({
            'NO': range(1, len(fea_name) + 1),
            'fea_ind': fea_idxes,
            'fea_name': fea_name,
            'ch': gp_info,
            'loc': locs,
        })
        df_all_eval = None
    else:
        df_all_eval, info = eval_6clf(slc_idx, fea_name, fea_idxes, locs, gp_info, data)
    info = pd.merge(info, sco, left_on='fea_ind', right_on='index').drop(['index'], axis=1, inplace=False)
    info.insert(loc=list(info.columns).index('abs_weight') + 1,
                column='abs_weight_normalize',
                value=info[['abs_weight']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0))

    sparsity = np.sum(info['abs_weight'] > 0) / len(info)

    return df_all_eval, info, sparsity


def eval_6clf(idx, fea_name, fea_ind, loc, ch_label, data):
    sheet_name = ['svm', 'lr', 'nb', 'knn', 'dt', 'rf']
    eval_name = ['acc', 'recall', 'precision', 'f1score', 'specificity']
    all_eval = np.zeros((len(sheet_name), len(idx), len(eval_name)))
    [X_train, Y_train, X_test, Y_test] = data

    for i in range(len(idx)):
        idx_tmp = idx[:(i + 1)]
        X_tr_tmp = X_train[:, idx_tmp]
        X_te_tmp = X_test[:, idx_tmp]

        # svm
        j = 0
        svm_clf = Pipeline([("scaler", StandardScaler()), ("svc_rbf", SVC(kernel="rbf"))])
        svm_clf.fit(X_tr_tmp, Y_train)
        pred = svm_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # lr
        j += 1
        oe = OrdinalEncoder()
        # oe.fit(Y_train.reshape(-1, 1)).categories_
        Y_train0 = oe.fit_transform(Y_train.reshape(-1, 1))
        lr_clf = LogisticRegression()
        lr_clf.fit(X_tr_tmp, Y_train0.reshape(-1))
        pred = lr_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # nb
        j += 1
        nb_clf = GaussianNB()
        nb_clf.fit(X_tr_tmp, Y_train)
        pred = nb_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # knn
        j += 1
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_tr_tmp, Y_train)
        pred = knn_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # dt
        j += 1
        dt_clf = DecisionTreeRegressor(random_state=2022)
        dt_clf.fit(X_tr_tmp, Y_train)
        pred = dt_clf.predict(X_te_tmp).astype(int)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # rf
        j += 1
        rf_clf = RandomForestClassifier(random_state=2022)
        rf_clf.fit(X_tr_tmp, Y_train)
        pred = rf_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

    # basic info
    info = pd.DataFrame({
        'NO': range(1, len(fea_name) + 1),
        'fea_ind': fea_ind,
        'fea_name': fea_name,
        'ch': ch_label,
        'loc': loc,
    })
    info_slc = info.iloc[idx, :].reset_index(drop=True, inplace=False)
    info_slc['NO'] = range(1, len(info_slc) + 1)

    df_all = {}
    for i in range(len(all_eval)):
        df = pd.concat([info_slc, pd.DataFrame(all_eval[i], columns=eval_name)], axis=1)
        df_all[sheet_name[i]] = df

    return df_all, info


def evaluate_pred(pred, true):
    acc = metrics.accuracy_score(true, pred)
    recall = metrics.recall_score(true, pred, average='macro')
    precision = metrics.precision_score(true, pred, average='macro')
    f1score = metrics.f1_score(true, pred, average='macro')
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    specificity = tn / (tn + fp)

    return [acc, recall, precision, f1score, specificity]


def set_seed(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    main()
