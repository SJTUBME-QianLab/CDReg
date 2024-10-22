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
    batch()


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


if __name__ == '__main__':
    main()

