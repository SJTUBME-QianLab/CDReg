import os
import pickle
import scipy
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

from LUAD import get_res_df, UpsetPlot, prepare_for_clf, PlotBar, DistHist


plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

eval_root = './WGBS/'
os.makedirs(eval_root, exist_ok=True)
data_name = 'CHR20p0.05gp20'
# data_root = f'./../data/{data_name}'
# result_root = f'./../results/{data_name}'
# test_root = f'./../results/testing/{data_name}'
data_root = '/home/data/public_data/Methyl/MethBank/HRA000099_pre/matrix_met_d20_chr22_na0.3_Mean'
result_root = f'/home/data/tangxl/ContrastSGL/results_appli/{data_name}'
test_root = f'/home/data/tangxl/ContrastSGL/results_appli/0811_add/results_{data_name}'

model_dict = {
    'LASSO': 'Lasso', 'Enet0.8': 'ENet', 'SGLASSO': 'SGLasso', 'pclogit': 'Pclogit',
    'dmp10': 'DMP', 'DMRcate': 'DMRcate',
    'L10.5L210.1Ls1.2Lc1_lr0.0001': 'CDReg',
}
model_dict_inv = {v: k for k, v in model_dict.items()}
colorsPal = sns.color_palette('Pastel1')
colors_dict = dict(zip(
    list(model_dict.values()),
    colorsPal[1:5] + ['#EBECA6', '#FFE9D2'] + [colorsPal[0]]
))


def main():
    base_name = os.path.basename(data_root)

    # sample information
    now = pd.read_csv(os.path.join(data_root, f'{base_name}{data_name}.csv'), index_col=0)
    info = pd.read_csv(os.path.join(data_root, '..', f'{base_name.rsplit("_", 1)[0]}_info.csv'), index_col=0)
    info.index = info['Sample ID']
    cut = info.loc[now.columns, :]
    pair = pd.read_csv(os.path.join(data_root, '..', f'{base_name.rsplit("_", 1)[0]}_pairDF.csv'), index_col=0)
    cut.loc[:, 'pairedTF'] = 0
    cut.loc[pair['Sample ID'], 'pairedTF'] = 1
    assert cut['pairedTF'].sum() == len(pair) == 324
    cut.to_csv(os.path.join(eval_root, 'sample_info.csv'))

    settle_results()

    fix = 14
    file_name = f'FigS3a_top{fix+1}_clf'
    seed_list = [14, 21, 24, 71, 73, 78, 79, 106, 110, 125]
    if not os.path.exists(os.path.join(eval_root, f'{file_name}.xlsx')):
        prepare_for_clf(file_name, fix, seed_list, model_dict, test_dir=test_root, eval_dir=eval_root)
    prepare_for_clf(file_name, fix, seed_list, model_dict, test_dir=test_root, eval_dir=eval_root)
    df = pd.read_excel(os.path.join(eval_root, f'{file_name}.xlsx'), sheet_name=None)
    PValues = pd.read_excel(os.path.join(eval_root, f'{file_name}.xlsx'), sheet_name='Ttest', index_col=0)
    color_dict = {vv: colors_dict[vv] for vv in model_dict.values()}
    config = {
        'size0': 2.2, 'size1': 4.8, 'padding': 40,
        'xlim': [0.68, 1.01], 'xticks': [0.7, 0.8, 0.9, 1.0], 'x0': 1.08,
    }
    PlotBar(df, 'acc', config, pv=PValues, name='FigS3a_BarSVMRF', color_dict=color_dict, eval_dir=eval_root)
    config = {
        'size0': 2.2, 'size1': 4.8, 'padding': 40,
        'xlim': [0.68, 1.01], 'xticks': [0.7, 0.8, 0.9, 1.0], 'x0': 1.08,
    }
    PlotBar(df, 'f1score', config, pv=PValues, name='FigS3a_BarSVMRF', color_dict=color_dict, eval_dir=eval_root)

    top = 15
    UpsetPlot(eval_root, model_dict.values(), 'FigS3b_Upset', top=top, unit='site')

    sites = pd.read_csv(os.path.join(data_root, base_name.rsplit("_", 1)[0].replace('matrix_met', 'sites') + f"_{data_name}.csv"))
    dist = []
    for gp in sites['gp_idx'].unique():
        dist = np.hstack([dist, np.diff(sites[sites['gp_idx']==gp]['start'].values)])
    DistHist(dist, name='FigS3c_DistHist', eval_dir=eval_root)


def settle_results():
    # Gene of all sites and all methods
    gene_info = pd.read_csv(os.path.join(data_root, f'sites_d20_chr22_na0.3_{data_name}.csv'))
    df_use = gene_info[['probe', 'gp_idx', '#chr', 'start', 'gene', 'pvalue', 'pFDR']]
    df_use.reset_index(drop=False, inplace=True)
    df_use.rename(columns={'index': 'fea_ind', 'probe': 'fea_name', 'gp_idx': 'group'}, inplace=True)
    df_rank = df_use.copy()
    for mm, mmn in model_dict.items():
        df0 = get_res_df(result_root, mm, 'final_weight')
        df1 = df0[['fea_ind', 'fea_name', 'ch', 'loc', 'abs_weight_normalize']]
        df1.rename(columns={'ch': 'group', 'loc': 'start', 'abs_weight_normalize': mmn}, inplace=True)
        df_use = pd.merge(df_use, df1, on=['fea_ind', 'fea_name', 'group', 'start'])
        df2 = df0[['fea_ind', 'fea_name', 'ch', 'loc', 'rank_min']]
        df2.rename(columns={'ch': 'group', 'loc': 'start', 'rank_min': mmn}, inplace=True)
        df_rank = pd.merge(df_rank, df2, on=['fea_ind', 'fea_name', 'group', 'start'])
        assert len(df_use) == len(df_rank) == len(df0)
    df_use.to_csv(os.path.join(eval_root, 'fea_slc.csv'), index=False)
    df_rank.to_csv(os.path.join(eval_root, 'fea_slc_rank.csv'), index=False)

    # top 15 vs rankings
    method_list = [kk for kk in model_dict.values() if kk not in ['DMRcate'] and 'w/o' not in kk]
    top_probes = df_use[['fea_name', '#chr', 'start', 'gene', 'pvalue', 'pFDR']]
    for mm in method_list:
        dfi = pd.concat([df_use[['fea_name', mm]], df_rank[[mm]]], axis=1)
        dfi.columns = ['fea_name', mm, mm+'_rank']
        dfi = dfi[dfi[mm+'_rank'] <= 15]
        top_probes = pd.merge(top_probes, dfi, on='fea_name', how='outer')
    top_probes.dropna(subset=[mm+'_rank' for mm in method_list], how='all', inplace=True)
    top_probes.to_csv(os.path.join(eval_root, 'Top15sitesAll.csv'), index=False)


if __name__ == '__main__':
    main()
