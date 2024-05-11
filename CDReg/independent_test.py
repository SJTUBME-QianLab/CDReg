import os
import numpy as np
import pandas as pd
import argparse
import random
from joblib import Parallel, delayed
# import seaborn
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--test_type', default='', type=str)
    args = parser.parse_args()

    data_name = args.data_name
    seed = args.seed
    test_type = args.test_type
    fold_num = args.fold_num

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    data_dir = './../data/'
    root_dir = './../results/'
    save0_dir = './../results/testing/'
    if data_name == 'LUAD':
        model_name_list = ['L10.5L210.2Ls1.2Lc0.3_lr0.001', 'LASSO', 'Enet0.8', 'SGLASSO']
        fold_num = 300
        folds_list = [88, 99, 115, 121, 150, 195, 204, 229, 246, 263]
    elif data_name == 'AD':
        model_name_list = ['L10.15L210.04Ls0.5Lc1.2_lr0.001', 'LASSO', 'Enet0.8', 'SGLASSO', 'pclogit']
        fold_num = 500
        folds_list = [13, 39, 214, 313, 337, 346, 398, 400, 443, 444]
    else:
        raise ValueError('data name')

    def sub_fold(kk):
        data, fea_info = get_data(os.path.join(data_dir, data_name, 'group_gene'),
                                  data_name=data_name, seed=seed, fold_num=fold_num, fold=kk, test_type=test_type)
        for model_name in model_name_list[:1]:
            new_save_dir = os.path.join(save0_dir, data_name, model_name,
                                        's'+str(seed)+'_'+test_type, 'fold'+str(kk))
            os.makedirs(new_save_dir, exist_ok=True)
            if os.path.isfile(os.path.join(new_save_dir, 'acc_allT.csv')):
                continue

            print(model_name, kk)
            if model_name in ['LASSO', 'Enet0.8', 'SGLASSO', 'pclogit']:
                save_dir = os.path.join(root_dir, data_name, '3m_default100_0.0001', model_name)
                sco = pd.read_csv(os.path.join(save_dir, '%s_coef_lam10_re1.csv' % model_name))
                if model_name == 'SGLASSO':
                    sco.iloc[:, 0] = [int(x) - 1 for x in sco.iloc[:, 0]]
                else:
                    sco = sco.iloc[1:, :]
                    sco.iloc[:, 0] = [int(x.split('V')[1]) - 1 for x in sco.iloc[:, 0]]
            else:
                save_dir = os.path.join(root_dir, data_name, model_name, 's'+str(seed))
                sco = pd.read_csv(os.path.join(save_dir, 'fea_scores.csv'))
                sco = sco.iloc[:, [0, sco.shape[1] - 1]]

            eval_clf_new(data, fea_info, sco, result_dir=new_save_dir, top_num=50)

    Parallel(n_jobs=2)(delayed(sub_fold)(fold) for fold in folds_list)


def get_data(data_dir, data_name, seed, fold_num=None, fold=None, test_type=None):
    Xall = np.load(os.path.join(data_dir, 'X_normL2.npy'))
    Yall = np.load(os.path.join(data_dir, 'Y.npy'))

    if data_name == 'LUAD':
        train_idx = list(range(492))
        test_idx = list(range(492, 492 + 183))
        fea_info = pd.read_csv(os.path.join(data_dir, 'info.csv')).iloc[1:, :]
        # IlmnID,gene_set,gene_num,UCSC_RefGene_Name,CHR,MAPINFO,gp_size,gp_idx

        if fold is not None and test_type == 'resample':
            assert len(Xall) == (492 + 183)
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

    elif data_name == 'AD':
        fea_info = pd.read_csv(os.path.join(data_dir, 'info.csv')).iloc[1:, :]
        if fold is not None and test_type == 'resample':
            resample_num = int(len(Yall) * 0.2)  # 200
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

    return data, fea_info


def eval_clf_new(data, fea_info, sco, result_dir, top_num=30):
    # df_all_eval = performance_app_cl(sco, data, fea_info, top_num=top_num)
    fea_name = fea_info['IlmnID'].values.tolist()
    fea_idxes = np.arange(len(fea_name))
    locs = fea_info['MAPINFO'].values.astype(float).astype(int)
    gp_info = fea_info['gp_idx'].values.astype(int)
    sco.columns = ['index', 'final_weight']
    sco.insert(loc=2, column='abs_weight', value=abs(sco['final_weight']))
    sco_rank = sco.sort_values(['abs_weight'], ascending=False, kind='mergesort').reset_index(drop=True)
    slc_idx = sco_rank['index'][:top_num]
    df_all_eval, info = eval_6clf(slc_idx, fea_name, fea_idxes, locs, gp_info, data)

    pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'final_%d_eval0.xlsx' % top_num))
    acc_TF = []
    for i, df in df_all_eval.items():
        df.to_excel(pd_writer, index=False, index_label=True, sheet_name=i)
        if i == 'svm':
            acc_TF.append(df.iloc[:, :list(df.columns).index('acc') + 1])
        else:
            acc_TF.append(df[['acc']])
    pd_writer.save()
    acc_TF = pd.concat(acc_TF, axis=1)
    acc_TF.columns = list(acc_TF.columns[:-(len(df_all_eval))]) + list(df_all_eval.keys())
    acc_TF.to_csv(os.path.join(result_dir, 'acc_allT.csv'), index=False, index_label=True)


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
    acc = sklearn.metrics.accuracy_score(true, pred)
    recall = sklearn.metrics.recall_score(true, pred, average='macro', zero_division=0)
    precision = sklearn.metrics.precision_score(true, pred, average='macro', zero_division=0)
    f1score = sklearn.metrics.f1_score(true, pred, average='macro', zero_division=0)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true, pred).ravel()
    specificity = tn / (tn + fp)

    return [acc, recall, precision, f1score, specificity]


if __name__ == '__main__':
    main()
