import numpy as np
import pandas as pd
import os
import time
from joblib import Parallel, delayed
from itertools import product
result_dir0 = './../results/'
eval_root = './../results/evaluation/'
vs_methods = ['LASSO', 'Enet0.8', 'SGLASSO', 'pclogit']


def main():
    # batch()

    prefixs = list(set([kk.split('_seed')[0] for kk in os.listdir(result_dir0) if kk.startswith('cov')]))
    select = pd.DataFrame()
    for pre in prefixs:
        data_seeds = [int(kk.split('_seed')[1]) for kk in sorted(os.listdir(result_dir0)) if kk.startswith(pre)]

        Parallel(n_jobs=10)(delayed(concat_my)(pre, seed=ss) for ss in data_seeds)

        concat_all(pre, data_seeds)

        packages = Parallel(n_jobs=10)(delayed(concat_ablation)(pre, seed=ss, mm=mm)
                                       for ss in data_seeds
                                       for mm in
                                       [kk for kk in sorted(os.listdir(os.path.join(result_dir0, f'{pre}_seed{ss}'))) if
                                        kk.startswith('L1') and 'Ls0Lc' not in kk and 'Lc0_lr' not in kk]
                                       )
        packages = [pp for pp in packages if pp is not None]
        print(len(packages))
        if len(packages) == 0:
            continue
        packages = pd.concat(packages, axis=0)
        packages['data'] = pre
        select = pd.concat([select, packages], axis=0)

    select.to_csv(os.path.join(result_dir0, 'concat', 'searchAbla.csv'), index=False)


def batch():
    df_gau1 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='Gau1-rep10')
    df_gau1 = df_gau1[['data_seed', 'method', 's', 'data']].dropna(axis=0, inplace=False)
    df_gau1[['data_seed', 's']] = df_gau1[['data_seed', 's']].astype(int)
    df_Lap2 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='Lap2-rep10')
    df_Lap2 = df_Lap2[['data_seed', 'method', 's', 'data']].dropna(axis=0, inplace=False)
    df_Lap2[['data_seed', 's']] = df_Lap2[['data_seed', 's']].astype(int)

    bd_gau1 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='bodong-Gau1')
    bd_gau1 = bd_gau1[['L1', 'L21', 'S', 'C', 'data_seed', 'method', 's', 'data']]
    pd.testing.assert_frame_equal(bd_gau1.iloc[:10, 4:7], df_gau1.iloc[:10, :3])
    bd_Lap2 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='bodong-Lap2')
    bd_Lap2 = bd_Lap2[['L1', 'L21', 'S', 'C', 'data_seed', 'method', 's', 'data']]
    pd.testing.assert_frame_equal(bd_Lap2.iloc[:10, 4:7], df_Lap2.iloc[:10, :3])

    df_ours = pd.concat([df_gau1.iloc[:30, :],df_Lap2.iloc[:30, :],
                         bd_gau1.iloc[10:, :],bd_Lap2.iloc[10:, :]],
                        axis=0).reset_index(drop=True)
    for i in range(len(df_ours)):
        tt = df_ours.loc[i, 'method']
        df_ours.loc[i, ['L1', 'L21', 'S', 'C']] = [
            float(tt.split('L1')[1].split('L21')[0]),
            float(tt.split('L21')[1].split('Ls')[0]),
            float(tt.split('Ls')[1].split('Lc')[0]),
            float(tt.split('Lc')[1].split('_lr')[0])
        ]

    lr, rate = 0.0001, 0.2
    cmd_list = []
    have, have_not = 0, 0
    for i in range(len(df_ours)):
        seed, method, s, data = df_ours.iloc[i, :4]
        dir_i = os.path.join(result_dir0, f'{data}_seed{seed}', method, f's{s}')
        if os.path.isfile(os.path.join(dir_i, 'eval_FS.xlsx')) and \
                os.stat(os.path.join(dir_i, 'eval_FS.xlsx')).st_size != 0:
            have += 1
        else:
            if os.path.exists(dir_i):
                print(dir_i)
                os.system('rm -r ' + dir_i)
            L1, L21, Ls, Lc = df_ours.iloc[i, 4:]
            cmd = 'python ./../CDReg/main_simu.py --seed %d --data_name %s ' \
                  '--L1 %f --L21 %f --Ls %f --Lc %f --lr %f' \
                  % (s, f'{data}_seed{seed}', L1, L21, Ls, Lc, lr)
            cmd_list.append(cmd)
            have_not += 1

    print(have_not, have, have + have_not)
    cmd_list = cmd_list[:2]
    half = int(len(cmd_list) / 2)
    cmd_list0 = ['CUDA_VISIBLE_DEVICES=0 ' + x for x in cmd_list[:half]]
    cmd_list1 = ['CUDA_VISIBLE_DEVICES=1 ' + x for x in cmd_list[half:]]
    Parallel(n_jobs=2)(delayed(device1)(ll, 1) for ll in [cmd_list0, cmd_list1])


def device1(cmd_list, nj=1):
    Parallel(n_jobs=nj)(delayed(os.system)(cmd) for cmd in cmd_list)


def cmdList():
    with open('./cmdList.txt', 'r') as f:
        data = f.read()
    cmd_list = data.split('\n')[:-1]
    print(len(cmd_list))

    half = int(len(cmd_list) / 2)
    cmd_list0 = ['CUDA_VISIBLE_DEVICES=0 ' + x for x in cmd_list[:half]]
    Parallel(n_jobs=3)(delayed(os.system)(cmd) for cmd in cmd_list0)



def concat_my(pre, seed):
    if os.path.isfile(os.path.join(result_dir0, 'concat', pre, f'seed{seed}.csv')):
        df_rand0 = pd.read_csv(os.path.join(result_dir0, 'concat', pre, f'seed{seed}.csv'))
        exist = [list(df_rand0.iloc[i, :3].values) for i in range(len(df_rand0))]
    else:
        df_rand0 = pd.DataFrame()
        exist = []

    head = ['data_seed', 'method', 's']
    data_name = f'{pre}_seed{seed}'
    df_rand = []
    method_list = [kk for kk in os.listdir(os.path.join(result_dir0, data_name)) if kk not in vs_methods]
    for mm in method_list:
        for s in [int(kk.split('s')[1]) for kk in os.listdir(os.path.join(result_dir0, data_name, mm))]:
            if [seed, mm, s] in exist:
                continue
            root_dir = os.path.join(result_dir0, data_name, mm, f's{s}')
            # print(root_dir)
            re = GetResults(root_dir)
            if not re.eval:
                continue
            df_rand.append(pd.concat([pd.DataFrame([seed, mm, s], index=head).T, re.onetime()], axis=1))
    method_list = [kk for kk in os.listdir(os.path.join(result_dir0, data_name)) if kk in vs_methods]
    for mm in method_list:
        for s in [int(kk.split('s')[1]) for kk in os.listdir(os.path.join(result_dir0, data_name, mm))]:
            if [seed, mm, s] in exist:
                continue
            root_dir = os.path.join(result_dir0, data_name, mm, f's{s}')
            re = GetResults(root_dir)
            if not re.eval:
                continue
            df_rand.append(pd.concat([pd.DataFrame([seed, mm, s], index=head).T, re.onetime()], axis=1))
    df_rand = pd.concat([df_rand0] + df_rand, axis=0)
    os.makedirs(os.path.join(result_dir0, 'concat', pre), exist_ok=True)
    df_rand.to_csv(os.path.join(result_dir0, 'concat', pre, f'seed{seed}.csv'), index=False)


def vs(main0, vv, cols, direct):
    if direct == 'high':
        return (main0[cols] > vv[cols]).all()
    elif direct == 'low':
        return (main0[cols] < vv[cols]).all()
    else:
        raise ValueError('wrong direct: ' + direct)


def concat_ablation(pre, seed, mm):
    all_df1 = pd.read_csv(os.path.join(result_dir0, 'concat', pre, f'seed{seed}.csv'))
    Ls = mm.split('Ls')[0] + 'Ls0Lc' + mm.split('Lc')[1]
    Lc = mm.split('Lc')[0] + 'Lc0_lr' + mm.split('_lr')[1]
    r0 = all_df1[all_df1['method'] == mm]
    rg = all_df1[all_df1['method'] == Ls]
    rc = all_df1[all_df1['method'] == Lc]
    rR = pd.concat([all_df1[all_df1['method'] == dd] for dd in vs_methods], axis=0)
    rR.loc['max'] = rR.max(axis=0)
    rR.loc['min'] = rR.min(axis=0)

    pack = pd.DataFrame()
    for i in range(len(r0)):
        main0 = r0.iloc[i, :]
        for j, k in product(range(len(rg)), range(len(rc))):
            gv = rg.iloc[j, :]
            cv = rc.iloc[k, :]

            g = vs(main0, cv, ['isol'], 'low') and vs(cv, gv, ['isol'], 'low')
            c = vs(main0, gv, ['indi_abs_w_mean'], 'low') and vs(gv, cv, ['indi_abs_w_mean'], 'low')

            met1 = ['acc', 'roc_auc']  # , 'precision', 'f1score'
            # met1 = ['acc', 'roc_auc', 'recall', 'precision', 'f1score', 'specificity']
            m1 = vs(main0, gv, met1, 'high') and vs(main0, cv, met1, 'high')
            met2 = ['isol', 'indi_abs_w_mean']
            # m2 = vs(main0, gv, met2, 'high') and vs(main0, cv, met2, 'high')
            mRa = vs(main0, rR.loc['max', :], met1, 'high')
            mRb = vs(main0, rR.loc['min', :], met2, 'low')
            if not (g and c and m1 and mRa and mRb):  # and tr
                continue

            dd = pd.concat([main0, gv, cv], axis=1).T
            pack = pd.concat([pack, dd, rR.iloc[:-2, :]], axis=0)
            # pack.to_csv(os.path.join(result_dir0, 'concat', pre, f'ablation_seed{seed}.csv'), index=False)

    if len(pack) == 0:
        return None
        # print(f'No satisfied -- {pre}, {seed}, {mm}')
    else:
        print(f'satisfied -- {pre}, {seed}, {mm}')
        print(pack)
        return pack


def concat_vsSOTA(pre, seed, mm):
    all_df1 = pd.read_csv(os.path.join(result_dir0, 'concat', pre, f'seed{seed}.csv'))
    r0 = all_df1[all_df1['method'] == mm]
    rR = pd.concat([all_df1[all_df1['method'] == dd] for dd in vs_methods], axis=0)
    rR.loc['max'] = rR.max(axis=0)
    rR.loc['min'] = rR.min(axis=0)

    pack = pd.DataFrame()
    for i in range(len(r0)):
        main0 = r0.iloc[i, :]

        met1 = ['roc_auc']  # , 'precision', 'f1score'
        # met1 = ['acc', 'roc_auc', 'recall', 'precision', 'f1score', 'specificity']
        met2 = ['isol', 'indi_abs_w_mean']
        # m2 = vs(main0, gv, met2, 'high') and vs(main0, cv, met2, 'high')
        mRa = vs(main0, rR.loc['max', :], met1, 'high')
        mRb = vs(main0, rR.loc['min', :], met2, 'low')
        if not (mRa and mRb):  # and tr
            continue

        pack = pd.concat([pack, r0.iloc[[i], :], rR.iloc[:-2, :]], axis=0)
        # pack.to_csv(os.path.join(result_dir0, 'concat', pre, f'vsSOTA_seed{seed}.csv'), index=False)

    if len(pack) == 0:
        return None
        # print(f'No satisfied -- {pre}, {seed}, {mm}')
    else:
        print(f'satisfied -- {pre}, {seed}, {mm}')
        print(pack)
        return pack


def concat_all(pre, data_seeds):
    dir_i = os.path.join(result_dir0, 'concat', pre)
    metrics_seeds = pd.DataFrame()
    for seed in data_seeds:
        if not os.path.isfile(os.path.join(dir_i, f'seed{seed}.csv')):
            continue
        df = pd.read_csv(os.path.join(dir_i, f'seed{seed}.csv'))
        metrics_seeds = pd.concat([metrics_seeds, df], axis=0)
    metrics_seeds.sort_values(['data_seed', 'method', 's'], inplace=True)
    metrics_seeds.to_csv(os.path.join(dir_i, 'search.csv'), index=False)


class GetResults:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        if os.path.isfile(os.path.join(self.root_dir, 'eval_FS.xlsx')):
            self.eval = True
        else:
            self.eval = False

    def acc_allT(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, 'eval_FS.xlsx'), sheet_name='acc_allT')
        return eval_FS

    def metrices(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, 'eval_FS.xlsx'), sheet_name='metrices')
        return eval_FS

    def group_check(self):
        evals = pd.read_excel(os.path.join(self.root_dir, 'eval_FS.xlsx'), sheet_name='group_check1')
        # inef = evals.iloc[0:30, -1].mean(axis=0)
        true = evals.iloc[30:48, -1].mean(axis=0)
        isol = evals.iloc[48:66, -1].mean(axis=0)

        evals = pd.read_excel(os.path.join(self.root_dir, 'eval_FS.xlsx'), sheet_name='group_check2')
        ineff = evals.iloc[[k + i * 22 for i in range(3) for k in range(19, 22)], -1].mean(axis=0)  # 8,9,10
        bias = evals.iloc[[k + i * 22 for i in range(3) for k in [7]], -1].mean(axis=0)  # 7
        group_mean = pd.DataFrame([true, isol, ineff, bias], index=['true', 'isol', 'inef', 'bias'])
        return group_mean.T

    def true_std(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, 'eval_FS.xlsx'), sheet_name='df_true_std')
        evals = eval_FS.iloc[:, -1].mean(axis=0)
        return pd.DataFrame({'true_std': [evals]})

    def onetime(self):
        return pd.concat([self.metrices(), self.group_check(), self.true_std()], axis=1)


if __name__ == '__main__':
    main()

