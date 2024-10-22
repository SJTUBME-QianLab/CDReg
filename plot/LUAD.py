import os
import pickle
import scipy
import pandas as pd
import numpy as np

import random
from joblib import Parallel, delayed
from itertools import product, permutations
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

from simulation import TTestPV, TTestPV_ind, plot_method_legend

plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

eval_root = './LUAD/'
os.makedirs(eval_root, exist_ok=True)
# data_root = './../data/LUAD/'
# result_root = './../results/LUAD/'
# test_root = './../results/testing/LUAD'
data_root = '/home/data/tangxl/ContrastSGL/casecontrol_data/LUAD'
result_root = '/home/data/tangxl/ContrastSGL/results_appli/LUAD/'
test_root = '/home/data/tangxl/ContrastSGL/results_appli/0811_add/results_LUAD/'

model_dict = {
    'LASSO': 'Lasso', 'Enet0.8': 'ENet', 'SGLASSO': 'SGLasso',
    'dmp10': 'DMP', 'DMRcate': 'DMRcate',
    'L10.5_L210.20_Lg0.0_Lcon0.3': 'CDReg w/o S',
    'L10.5_L210.20_Lg1.2_Lcon0.0': 'CDReg w/o C',
    'L10.5_L210.20_Lg1.2_Lcon0.3': 'CDReg',
    # 'L10.5L210.2Ls0Lc0.3_lr0.001': 'CDReg w/o S',
    # 'L10.5L210.2Ls1.2Lc0_lr0.001': 'CDReg w/o C',
    # 'L10.5L210.2Ls1.2Lc0.3_lr0.001': 'CDReg',
}
model_dict_inv = {v: k for k, v in model_dict.items()}
colorsPal = sns.color_palette('Pastel1')
colors_dict = dict(zip(
    list(model_dict.values()),
    colorsPal[1:4] + ['#EBECA6', '#FFE9D2'] + colorsPal[6:8] + [colorsPal[0]]
))
metric_dict = {
    'acc': 'Accuracy', 'auc': 'AUC', 'f1score': 'F1-score', 'AUPRC': 'AUPRC',
    'svm': 'Support Vector Machine', 'lr': 'Logistic Regession', 'rf': 'Random Forest',
    'pair': 'Paired', 'indi': 'Individual',
    'cancer': 'Cancer', 'normal': 'Normal',
}


def main():
    settle_results()

    # from Top25sites.csv; add 'GENE' column using NCBI searching : https://www.ncbi.nlm.nih.gov/gene/?term=
    data = pd.read_excel('./Results_LUAD.xlsx', sheet_name='Top25sites')
    data['logP'] = - np.log10(data['padj'])
    DotWeightPvalue(data, name='Fig4a_Top25sites')

    fix = 24
    file_name = f'Fig4b_top{fix+1}_clf'
    folds = [88, 99, 115, 121, 150, 195, 204, 229, 246, 263]
    if not os.path.exists(os.path.join(eval_root, f'{file_name}.xlsx')):
        prepare_for_clf(file_name, fix, folds, model_dict)
    df = pd.read_excel(os.path.join(eval_root, f'{file_name}.xlsx'), sheet_name=None)
    PValues = pd.read_excel(os.path.join(eval_root, f'{file_name}.xlsx'), sheet_name='Ttest', index_col=0)
    config = {
        'size0': 2.2, 'size1': 4, 'padding': 40,
        'xlim': [0.66, 1.02], 'xticks': [0.7, 0.8, 0.9, 1.0], 'x0': 1.05,
    }
    PlotBar(df, 'acc', config, pv=PValues, name='Fig4b_BarSVMRF')
    config = {
        'size0': 2.16, 'size1': 4, 'padding': 36,
        'xlim': [0.2, 0.92], 'xticks': [0.2, 0.5, 0.8], 'x0': 1.00,
    }
    PlotBar(df, 'f1score', config, pv=PValues, name='Fig4b_BarSVMRF')

    model_name = model_dict_inv['CDReg']
    norm = 'MinMax'
    if not os.path.exists(os.path.join(eval_root, f'{model_name}_49_PCA.pkl')):
        prepare_for_startend(model_name)
    with open(os.path.join(eval_root, f'{model_name}_0_PCA.pkl'), 'rb') as f:
        start_PCA = pickle.load(f)
    with open(os.path.join(eval_root, f'{model_name}_49_PCA.pkl'), 'rb') as f:
        end_PCA = pickle.load(f)

    StartEndPCA(start_PCA, end_PCA, 'Fig4d_PCAStartEnd')

    dist_start, _ = CalDistancesGlobal(start_PCA, dist='EucXY', Norm=norm, simple=True)
    dist_end, _ = CalDistancesGlobal(end_PCA, dist='EucXY', Norm=norm, simple=True)
    dist_start.insert(loc=0, column='Stage', value='Start')
    dist_end.insert(loc=0, column='Stage', value='End')
    dataVS_PCA = pd.concat([dist_start, dist_end], axis=0)
    # DistanceStartEnd(dataVS_PCA, name='Fig4e_StartEndPCABar')
    save = pd.merge(
        dataVS_PCA.groupby(['Stage', 'group'], as_index=False).mean().rename(columns={'value': 'Mean'}),
        dataVS_PCA.groupby(['Stage', 'group'], as_index=False).std().rename(columns={'value': 'Std'}),
    )
    save['Count'] = dataVS_PCA.groupby(['Stage', 'group']).size().values
    save['Standard error'] = save['Std'] / np.sqrt(save['Count'])
    save.to_csv(os.path.join(eval_root, 'Fig4e_StartEndPCABar.csv'), index=False)

    start, end = prepare_for_startend_cos(model_name)
    for dist in ['L1', 'Euc', 'Cosine', 'Dot']:
        dist_start, _ = CalDistancesGlobal(start, dist=dist, Norm=norm, simple=True)
        dist_end, _ = CalDistancesGlobal(end, dist=dist, Norm=norm, simple=True)
        dist_start.insert(loc=0, column='Stage', value='Start')
        dist_end.insert(loc=0, column='Stage', value='End')
        dataVS = pd.concat([dist_start, dist_end], axis=0)
        DistanceStartEnd(dataVS, name=f'Fig4e_StartEnd{dist}')

    file_name = 'Fig4f_cluster'
    if not os.path.exists(os.path.join(eval_root, f'{file_name}_data0.csv')):
        prepare_for_cluster(file_name)
    raw_use = pd.read_csv(os.path.join(eval_root, f'{file_name}_data0.csv'))
    df_use = pd.read_csv(os.path.join(eval_root, f'{file_name}_weight0.csv'))
    LUADCluster(raw_use, df_use, name=file_name, pv='padj')

    if not os.path.exists(os.path.join(eval_root, 'AdjCorr.pkl')):
        prepare_for_corr()
    with open(os.path.join(eval_root, 'AdjCorr.pkl'), 'rb') as f:
        DistancesAll = pickle.load(f)
    Corr_raw = Dist2Corr(DistancesAll)
    ScatterDistCorr(Corr_raw.copy(), name='FigS1a_ScatterDistCorr', cut=5050, frac=0.01)
    DistHist(Corr_raw['dist'], name='FigS1b_DistHist')

    if not os.path.exists(os.path.join(eval_root, 'raw_PCA.pkl')):
        prepare_for_pca()
    with open(os.path.join(eval_root, 'raw_PCA.pkl'), 'rb') as f:
        save1 = pickle.load(f)

    data = save1['data']
    ScatterBoxHue4(data.copy(), pv='T', name='FigS1c_PCABox')
    data.to_csv(os.path.join(eval_root, 'FigS1c_PCABox.csv'), index=False)

    dist, _ = CalDistancesGlobal(data, Norm=None, simple=False)
    DistanceSnsH(dist.copy(), name='FigS1d_PCABar')
    save = pd.merge(
        dist.groupby(['group'], as_index=False).mean().rename(columns={'value': 'Mean'}),
        dist.groupby(['group'], as_index=False).std().rename(columns={'value': 'Std'}),
    )
    save['Count'] = dist.groupby(['group']).size().values
    save['Standard error'] = save['Std'] / np.sqrt(save['Count'])
    save.to_csv(os.path.join(eval_root, 'FigS1d_PCABar.csv'), index=False)

    # top 25 vs rankings
    df_use = pd.read_csv(os.path.join(eval_root, 'fea_slc.csv'))
    df_rank = pd.read_csv(os.path.join(eval_root, 'fea_slc_rank.csv'))
    method_list = [kk for kk in model_dict.values() if kk not in ['Pclogit', 'DMRcate'] and 'w/o' not in kk]
    top_probes = df_use[['fea_name', 'CHR', 'MAPINFO', 'gene_set']]
    for mm in method_list:
        dfi = pd.concat([df_use[['fea_name', mm]], df_rank[[mm]]], axis=1)
        dfi.columns = ['fea_name', mm, mm+'_rank']
        dfi = dfi[dfi[mm+'_rank'] <= 25]
        top_probes = pd.merge(top_probes, dfi, on='fea_name', how='outer')
    top_probes.dropna(subset=[mm+'_rank' for mm in method_list], how='all', inplace=True)
    top_probes.rename(columns={'fea_name': 'IlmnID'}, inplace=True)
    probe_pvals = pd.read_csv(os.path.join(eval_root, 'probe_pvals.csv'))
    top_probes = pd.merge(top_probes, probe_pvals, on='IlmnID', how='left')
    top_probes.to_csv(os.path.join(eval_root, 'Fig4_Top25sitesVS.csv'), index=False)

    method_list = [kk for kk in model_dict.values() if kk not in ['Pclogit'] and 'w/o' not in kk]
    for top in [25, 50]:
        UpsetPlot(eval_root, method_list, 'FigS2_Upset2', top=top, unit='site')
        UpsetPlot(eval_root, method_list, 'FigS2_Upset2', top=top, unit='gene')

    # segments for example
    df_use = pd.read_csv(os.path.join(eval_root, 'fea_slc.csv'))
    df_use.sort_values('fea_ind', inplace=True)
    df_use.rename(columns={'fea_name': 'IlmnID'}, inplace=True)
    examples = [
        [17054, 17079],
        [44276, 44285],
        [10521, 10540],
        [47075, 47079],
    ]
    eg_seg = dict()
    for locs in examples:
        eg_seg['-'.join(str(kk) for kk in locs)] = segment(df_use.copy(), locs, head='IlmnID', dmr='Fisher')

    pd_writer = pd.ExcelWriter(os.path.join(eval_root, 'TableS4_segments.xlsx'))
    for locs, df in eg_seg.items():
        df.to_excel(pd_writer, index=False, sheet_name=locs)
    pd_writer.save()


def get_res_df(root_dir, mm, sheet_name, fold=None):
    if mm.startswith('L1'):
        if 'CHR' in root_dir:
            res_dir = os.path.join(root_dir, mm, 's1')
        else:
            res_dir = os.path.join(root_dir, mm)
    elif mm in ['LASSO', 'Enet0.8', 'SGLASSO', 'pclogit']:
        res_dir = os.path.join(root_dir, '3m_default100_0.0001', mm)
    elif mm in ['dmp10', 'DMRcate']:
        res_dir = os.path.join(root_dir, 'other', mm)
    else:
        raise ValueError(mm)

    if sheet_name in ['acc_allT', 'final_weight']:
        df = pd.read_excel(os.path.join(res_dir, 'eval_FS1.xlsx'), sheet_name=sheet_name)
    elif sheet_name in ['svm', 'rf', 'lr']:
        assert fold is not None
        df = pd.read_excel(os.path.join(res_dir, 's1_resample', f'fold{fold}', 'final_50_eval1.xlsx'), sheet_name=sheet_name)
    else:
        raise ValueError(sheet_name)
    return df


def settle_results():
    # Gene of all sites and all methods
    gene_info = pd.read_csv(os.path.join(data_root, 'group_gene', 'info.csv'))
    gene_info = gene_info.drop(0, axis=0).reset_index(drop=True)
    df_use = gene_info[['IlmnID', 'gp_idx', 'CHR', 'MAPINFO', 'gene_set']]
    df_use.reset_index(drop=False, inplace=True)
    df_use.rename(columns={'index': 'fea_ind', 'IlmnID': 'fea_name', 'gp_idx': 'group'}, inplace=True)
    df_rank = df_use.copy()
    for mm, mmn in model_dict.items():
        df0 = get_res_df(result_root, mm, 'final_weight')
        df1 = df0[['fea_ind', 'fea_name', 'ch', 'loc', 'abs_weight_normalize']]
        df1.rename(columns={'ch': 'group', 'loc': 'MAPINFO', 'IlmnID': 'fea_name', 'abs_weight_normalize': mmn}, inplace=True)
        df_use = pd.merge(df_use, df1, on=['fea_ind', 'fea_name', 'group', 'MAPINFO'])
        df2 = df0[['fea_ind', 'fea_name', 'ch', 'loc', 'rank_min']]
        df2.rename(columns={'ch': 'group', 'loc': 'MAPINFO', 'IlmnID': 'fea_name', 'rank_min': mmn}, inplace=True)
        df_rank = pd.merge(df_rank, df2, on=['fea_ind', 'fea_name', 'group', 'MAPINFO'])
        assert len(df_use) == len(df_rank) == len(df0)
    df_use.to_csv(os.path.join(eval_root, 'fea_slc.csv'), index=False)
    df_rank.to_csv(os.path.join(eval_root, 'fea_slc_rank.csv'), index=False)

    # P-value for all sites
    if not os.path.exists(os.path.join(eval_root, 'probe_pvals.csv')):
        pvals = pd.read_csv(os.path.join(data_root, 'part_data', 'pvalues.csv'), index_col=0)
        raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
        probes = raw[['IlmnID']][2:].reset_index(drop=True)
        probe_pvals = pd.concat([probes, pvals.iloc[2:, :].reset_index(drop=True)], axis=1)
        print(len(raw), len(pvals), len(probe_pvals))  # 373615 373615 373613
        probe_pvals.to_csv(os.path.join(eval_root, 'probe_pvals.csv'), index=False)

    # Top 25 sites with P-value and weight
    probe_pvals = pd.read_csv(os.path.join(eval_root, 'probe_pvals.csv'))
    top_probes = df_use[['fea_name', 'CHR', 'MAPINFO', 'gene_set', 'CDReg']]
    top_probes = pd.merge(top_probes, df_rank[['fea_name', 'CDReg']].rename(columns={'CDReg': 'Rank'}), on='fea_name')
    top_probes.sort_values('Rank', ascending=True, inplace=True)
    top_probes.rename(columns={'fea_name': 'IlmnID'}, inplace=True)
    top_probes = pd.merge(top_probes.iloc[:25, :], probe_pvals, on='IlmnID', how='left')
    top_probes.to_csv(os.path.join(eval_root, 'Top25sites.csv'), index=False)

    # w OR w/o Contrast Scheme
    ours = get_res_df(result_root, model_dict_inv['CDReg'], 'acc_allT')
    ours = ours[['fea_ind', 'fea_name']]
    ours.columns = ['ours_ind', 'ours_name']
    ours['ours_rank'] = np.arange(1, 51)
    Lcon = get_res_df(result_root, model_dict_inv['CDReg w/o C'], 'acc_allT')
    Lcon = Lcon[['fea_ind', 'fea_name']]
    Lcon.columns = ['Lcon_ind', 'Lcon_name']
    Lcon['Lcon_rank'] = np.arange(1, 51)
    concat = pd.merge(ours, Lcon, left_on='ours_name', right_on='Lcon_name', how='outer')
    union = [(concat.loc[i, 'ours_name'] if (concat.loc[i, 'ours_name'] is not np.nan)
              else concat.loc[i, 'Lcon_name'])
             for i in range(len(concat))]
    concat.rename(columns={'ours_name': 'IlmnID'}, inplace=True)
    concat['IlmnID'] = union
    concat.drop(['Lcon_name', 'ours_ind', 'Lcon_ind'], axis=1, inplace=True)
    concat = pd.merge(concat, gene_info[['IlmnID', 'CHR', 'MAPINFO', 'gene_set']], on='IlmnID')
    concat.to_csv(os.path.join(eval_root, 'Lcon_merge.csv'), index=False)


def prepare_for_cluster(file_name):
    df = pd.read_csv(os.path.join(eval_root, 'fea_slc.csv'))
    start, end = np.where(df['fea_name'] == 'cg21585409')[0][0], np.where(df['fea_name'] == 'cg26660312')[0][0]
    print(start, end)
    df_use = df.iloc[start:end + 1, :][['fea_name', 'CHR', 'MAPINFO', 'CDReg']]

    probe_pvals = pd.read_csv(os.path.join(eval_root, 'probe_pvals.csv'))

    df_use = pd.merge(df_use, probe_pvals, left_on='fea_name', right_on='IlmnID').drop(['fea_name'], axis=1)
    df_use.to_csv(os.path.join(eval_root, f'{file_name}_weight0.csv'), index=False)

    raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
    raw_use = pd.merge(raw, df_use[['IlmnID']], on='IlmnID')
    raw_use = raw_use.iloc[:, :(3 + 492)]
    raw_use.loc[-1] = ['-1', -1, -1] + [0] * 460 + [1] * 32
    raw_use.index = raw_use.index + 1
    raw_use.sort_index(inplace=True)
    raw_use.to_csv(os.path.join(eval_root, f'{file_name}_data0.csv'), index=False)


def DotWeightPvalue(data, name=''):
    colors_gene = {
        'Kir2.2': '#F2AD60', 'Kv1.1': '#4193C5', 'KCNQ1DN': '#BCB616', 'EGFR': '#C84112', 'CASC15': '#40478C',
    }
    colors = dict(zip(range(1, 26), [colors_gene[data.loc[i, 'GENE']] for i in range(25)]))

    fig = plt.figure(figsize=(6.4/2.54, 17.2/2.54), facecolor='none')
    gs = fig.add_gridspec(
        1, 2,  width_ratios=(1, 1), wspace=0.05
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    a1 = sns.stripplot(
        ax=ax1, data=data, orient='h', x='logP', y='Rank', hue='Rank',
        palette=colors, edgecolor='black', linewidth=0.5, #legend=False,
    )
    ax1.set_xlabel('-log(P)', fontsize=12)
    ax1.set_xlim(-5, 88)
    ax1.set_xticks([0, 40, 80])

    a2 = sns.stripplot(
        ax=ax2, data=data, orient='h', x='Weight', y='Rank', hue='Rank',
        palette=colors, edgecolor='black', linewidth=0.5, #legend=False,
    )
    ax2.set_xlabel('Weight', fontsize=12)
    ax2.set_xlim(0.55, 1.05)
    ax2.set_xticks([0.6, 0.8, 1.0])
    
    for ax in [ax1, ax2]:
        ax.tick_params(
            axis='both', labelsize=10, length=2, pad=0,
            bottom=False, top=True, left=False, right=False,
            labelbottom=False, labeltop=True, labelleft=False, labelright=False,
        )
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('')
        ax.grid(axis='y', linestyle=':')
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.legend_.remove()

    fig.savefig(os.path.join(eval_root, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def LUADCluster(raw_use, df_use, name, pv='pvals'):
    random.seed(2024)
    os.environ['PYTHONHASHSEED'] = str(2024)
    np.random.seed(2024)

    raw_use = raw_use.iloc[:-1, :]
    df_use = df_use.iloc[:-1, :]
    offset = 5270480
    locs = df_use.loc[:, 'MAPINFO'] - offset
    sample_num = 20
    label = raw_use.iloc[0, :]

    data_need_c = raw_use.iloc[1:, [0, 1, 2] + np.where(label == 0)[0].tolist()]
    c_ave = np.mean(data_need_c.iloc[:, 3:].values, axis=1)
    samples_c = random.sample(list(range(3, data_need_c.shape[1])), sample_num)
    data_need_n = raw_use.iloc[1:, [0, 1, 2] + np.where(label == 1)[0].tolist()]
    n_ave = np.mean(data_need_n.iloc[:, 3:].values, axis=1)
    samples_n = random.sample(list(range(3, data_need_n.shape[1])), sample_num)

    data = []
    for idx in samples_c:
        dfi = pd.DataFrame({
            'loc': locs,
            'value': data_need_c.iloc[:, idx].values,
            'label': 'Cancer',
        })
        data.append(dfi)
    for idx in samples_n:
        dfi = pd.DataFrame({
            'loc': locs,
            'value': data_need_c.iloc[:, idx].values,
            'label': 'Normal',
        })
        data.append(dfi)
    data = pd.concat(data, axis=0)

    def set_axis(ax1):
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['right'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['top'].set_linewidth(1.5)

    ax_idx = [2]
    xlim = 540
    area = 12
    marker = 'o'
    
    # ################# divide subplots
    from matplotlib import gridspec
    import matplotlib.ticker as mtick

    fig = plt.figure(figsize=(5, 3.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 2], hspace=0.1)
    [ax1, ax2] = [plt.subplot(gs[i]) for i in [0, 1]]

    # ################# ax1: scatter
    ax1.scatter(
        x=data[data['label'] == 'Normal']['loc'], 
        y=data[data['label'] == 'Normal']['value'],
        s=area, marker=marker,
        color='cornflowerblue', label='Normal',
    )
    ax1.scatter(
        x=data[data['label'] == 'Cancer']['loc'], 
        y=data[data['label'] == 'Cancer']['value'],
        s=area, marker=marker, 
        color='salmon', label='Cancer',
    )
    ax1.plot(locs, n_ave,
             linewidth=1.2, 
             color='royalblue', label='Normal average'
    )
    ax1.plot(locs, c_ave,
             linewidth=1.2,
             color='crimson', label='Cancer average'
    )
    ax1.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=False, top=False, left=True, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )
    set_axis(ax1)
    ax1.set_xlim(0, xlim)
    ax1.grid(axis='x', linewidth=0.8, linestyle=':')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylabel('Methylation level', fontsize=12)

    # ################# ax2: p-value
    ins1 = ax2.plot(locs, - np.log10(df_use[pv]),
        color='seagreen', label='P-value', linewidth=1.2, marker='^', markersize=4
    )
    ax2.set_xlim(0, xlim)
    ax2.grid(axis='x', linewidth=0.8, linestyle=':')
    ax2.set_ylim(2, 11)
    ax2.set_yticks(ticks=[3, 6, 9])
    # ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax2.set_ylabel('-log(P)', fontsize=12)
    ax2.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    set_axis(ax2)

    # ################# ax2.twinx: weight
    ax3 = ax2.twinx()
    ins2 = ax3.plot(locs, df_use['CDReg'],
        color='navy', label='Weight', linewidth=1.2, marker='.'
    )
    ax3.set_ylim(-0.7/7*0.24, 0.24)
    ax3.yaxis.set_tick_params(labelsize=12)
    ax3.set_yticks(ticks=[0.0, 0.1, 0.2])
    ax3.set_ylabel('Weight', fontsize=12)
    ax3.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=False, right=True,
        labelbottom=False, labeltop=False, labelleft=False, labelright=True,
    )

    # ################# legend
    ax1.legend(
        loc='lower center', fontsize=10, # borderaxespad=0.2, bbox_to_anchor=(1.05, 0.02),
        ncol=2, handlelength=1.0, columnspacing=0.5, handletextpad=0.2, borderpad=0.5
    )
    lns = ins1 + ins2
    labs = [l.get_label() for l in lns]
    ax2.legend(
        lns, labs, fontsize=10,
        # loc='lower center', bbox_to_anchor=(0.36, -0.05),
        loc='center',
        ncol=2, handlelength=1.0, columnspacing=0.5, handletextpad=0.2, borderpad=0.5
    )
    
    # ################# x-label
    locs0 = locs.copy()
    locs0 = locs0.astype(int)
    locs0.loc[ax_idx] = ''
    ax1.set_xticks(ticks=locs)
    ax2.set_xticks(ticks=locs, labels=locs0)
    ax2.text(0.5, -0.6, 'CHR 7, Location (+%d)\n' % offset,
             horizontalalignment='center', verticalalignment='bottom', fontsize=12, transform=ax2.transAxes)

    fig.savefig(os.path.join(eval_root, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()
    save = pd.DataFrame({
        'loc-offset': locs,
        'Normal average': n_ave,
        'Cancer average': c_ave,
        'P-value': df_use[pv],
        '-log(P)': - np.log10(df_use[pv]),
        'Weight': df_use['CDReg'],
    })
    save = pd.concat([raw_use.iloc[1:, :3].reset_index(drop=True), save], axis=1)
    save.to_csv(os.path.join(eval_root, f'{name}_weight.csv'), index=False)
    data.to_csv(os.path.join(eval_root, f'{name}_data.csv'), index=False)


def get_average(clf, metric, mm_dict, folds, root_dir):
    evals_folds = []
    for fold in folds:
        evals_all = dict()
        for mm, mmn in mm_dict.items():
            df = pd.read_excel(os.path.join(root_dir, mm, 's1_resample', f'fold{fold}', 'final_50_eval0.xlsx'), sheet_name=clf)
            evals_all[mmn] = df[metric].values
        evals_all = pd.DataFrame(evals_all)  # column=50, row=methods
        evals_folds.append(evals_all)  # ten-times repeat

    folds = np.stack([dd.values for dd in evals_folds], axis=0)  # 10*50*methods
    folds_ave = pd.DataFrame(folds.mean(axis=0), columns=evals_folds[0].columns)  # 50*methods，10-time average
    folds_std = pd.DataFrame(folds.std(axis=0), columns=evals_folds[0].columns)  # 50*methods，10-time std
    return evals_folds, folds_ave, folds_std


def get_pval(folds_1site, metric_name, mm_dict):
    metric = metric_name.split('_')[1]
    if metric in ['acc', 'auc', 'f1score']:
        side = 'greater'
    elif metric in ['indi', 'isol']:
        side = 'less'
    else:
        raise ValueError(metric)
    table = pd.DataFrame()
    ours = folds_1site.loc[:, 'CDReg']
    for method in mm_dict.values():
        if method in ['CDReg']:
            continue
        pp = TTestPV(ours, folds_1site.loc[:, method], side)
        table.loc[metric_name, method] = pp
    return table


def prepare_for_clf(name, fix, folds, mm_dict, test_dir=test_root, eval_dir=eval_root):
    save_folds_1site = dict()
    save_pv_Ttest = []
    for clf in ['svm', 'rf']:
        for metric in ['acc', 'f1score']:
            evals_folds, folds_ave, _ = get_average(clf, metric, mm_dict, folds, test_dir)  # 指定分类器、指定指标
            folds_1site = pd.DataFrame(np.stack([dd.iloc[fix, :].values for dd in evals_folds], axis=0),
                                       columns=evals_folds[0].columns)
            metric_name = '_'.join([clf, metric])
            save_folds_1site[metric_name] = folds_1site
            save_pv_Ttest.append(get_pval(folds_1site, metric_name, mm_dict))  # 1*方法数，df格式，index=metric_name

    df_pv_Ttest = pd.concat(save_pv_Ttest, axis=0)

    pd_writer = pd.ExcelWriter(os.path.join(eval_dir, f'{name}.xlsx'))
    for kk, df in save_folds_1site.items():
        df.to_excel(pd_writer, index=False, index_label=True, sheet_name=kk)
    df_pv_Ttest.to_excel(pd_writer, index=True, index_label=True, sheet_name='Ttest')
    pd_writer.save()


def PlotBar(df, met, config, pv=None, name='', color_dict=None, eval_dir=eval_root):
    if color_dict is None:
        color_dict = colors_dict
    method_list = list(color_dict.keys())

    def SnsConcatMetric(mt):
        data = []
        for clf in ['svm', 'rf']:
            dfi = df[f'{clf}_{mt}']
            for mm in method_list:
                data.append(pd.DataFrame({
                    'idx': range(10),
                    'clf': clf,
                    'method': mm,
                    mt: dfi[mm],
                }))
        data = pd.concat(data, axis=0)
        return data

    data = pd.merge(SnsConcatMetric('acc'), SnsConcatMetric('f1score'))
    fig = plt.figure(figsize=(config['size0'], config['size1']))
    ax1 = plt.subplot(1, 1, 1)

    bar1 = sns.barplot(
        ax=ax1, data=data, y='clf', x=met, orient='h', errorbar='se',
        order=['svm', 'rf'],
        hue='method', hue_order=method_list, palette=color_dict, width=0.85,
    )
    str1 = sns.stripplot(
        ax=ax1, data=data, y='clf', x=met, orient='h',
        order=['svm', 'rf'],
        hue='method', hue_order=method_list, dodge=True,
        palette=['grey']*len(method_list), size=4, edgecolor="0.2", linewidth=0.5, alpha=0.5
    )

    ax1.set_xlim(*config['xlim'])
    ax1.set_xticks(config['xticks'])
    ax1.set_xlabel('')

    ax1.set_ylim(1.5, -0.5)
    ax1.set_ylabel('')
    if met == 'acc':
        ax1.set_yticks(ticks=[0, 1], labels=['SVM', 'RF'], 
                       rotation=90, verticalalignment='center')
    else:
        ax1.set_yticks(ticks=[0, 1], labels=['', ''],)
    ax1.set_title(metric_dict[met], size=12, weight='bold')

    ax1.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )

    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_visible(False)
    
    ax1.set_facecolor('None')  # 透明背景
    for i in bar1.containers:
        bar1.bar_label(i, fmt='%.3f', padding=-config['padding'], fontsize=10)

    if pv is not None:
        x0 = config['x0']
        start = ax1.patches[0].get_y()
        width = ax1.patches[2].get_y() - ax1.patches[0].get_y()
        ax1.text(x0, start - width * 0.5, 'P-value',
                 horizontalalignment='center', verticalalignment='center')
        ax1.text(x0, start - width * 0.5 + 1, 'P-value',
                 horizontalalignment='center', verticalalignment='center')
        ax1.text(x0, start + 0.02, '-'*12,
                 horizontalalignment='center', verticalalignment='bottom')
        ax1.text(x0, start + 1.02, '-'*12,
                 horizontalalignment='center', verticalalignment='bottom')
        for i, mm in enumerate(method_list[:-1]):
            ax1.text(x0, start + width * (i+0.5), 
                     '{:.2e}'.format(pv.loc[f'svm_{met}', mm]),
                    horizontalalignment='center', verticalalignment='center')
            ax1.text(x0, start + width * (i+0.5) + 1, 
                     '{:.2e}'.format(pv.loc[f'rf_{met}', mm]),
                    horizontalalignment='center', verticalalignment='center')

    if met == 'acc':
        plot_method_legend(color_dict, os.path.join(eval_dir, f'{name}_legend_rows.svg'))
        plot_method_legend(color_dict, os.path.join(eval_dir, f'{name}_legend_cols.svg'), cols=-1)
        # ax1.legend(
        #     loc='center right', fontsize=10, bbox_to_anchor=(-0.15, 0.5), # borderaxespad=0.2,
        #     ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
        # )
    ax1.legend_.remove()

    SaveName = '{}_{}_{}'.format(name, met, '' if pv is None else 'p')
    fig.savefig(os.path.join(eval_dir, f'{SaveName}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def AdjCorr(ch, raw):
    dfi = raw[raw['CHR'] == ch].reset_index(drop=True)
    data_c = dfi.iloc[:, 3:(3+460)].values.astype(float)
    data_n = dfi.iloc[:, (3+460):(3+492)].values.astype(float)
    info = dfi.iloc[:, :3]
    num = len(dfi)
    del dfi
    res = {'info': info}

    dist_i = []
    for i in range(num - 1):
        dist_i.append(np.corrcoef(data_c[i, :], data_c[i+1, :])[0, 1])
    res['corr_cancer'] = np.array(dist_i)
    dist_i = []
    for i in range(num - 1):
        dist_i.append(np.corrcoef(data_n[i, :], data_n[i+1, :])[0, 1])
    res['corr_normal'] = np.array(dist_i)
    return res


def prepare_for_corr():
    raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
    raw = raw.iloc[2:, :(3+492)].reset_index(drop=True)
    raw[['CHR', 'MAPINFO']] = raw[['CHR', 'MAPINFO']].astype(float).astype(int)

    DistancesAll = dict()
    for i in range(1, 23):
        DistancesAll[i] = AdjCorr(i, raw)

    assert sum([len(kk['corr_cancer']) + 1 for kk in DistancesAll.values()]) == len(raw)

    with open(os.path.join(eval_root, 'AdjCorr.pkl'), 'wb') as f:
        pickle.dump(DistancesAll, f)


def ScatterDistCorr(Corr, name='', cut=0, frac=1):
    ss = np.std(Corr['dist'])
    Corr['expD'] = np.exp(-Corr['dist'] ** 2 / (2 * ss ** 2))
    Corr['corrNAbs'] = abs(Corr['corrN'])
    Corr['corrCAbs'] = abs(Corr['corrC'])

    colors_nc = {'Cancer': '#E6A8A5', 'Normal': '#CBCBCB'}  # red, gray
    colors_text = {'Cancer': '#99302B', 'Normal': '#676767'}  # red, gray
    dot_size = 2
    
    fig = plt.figure(figsize=(4.4, 3.3), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)
    
    data0 = Corr.copy()
    if cut > 0:
        data0 = data0[data0['dist'] <= cut].reset_index(drop=True)
    if frac < 1:
        data0 = data0.sample(frac=frac, random_state=2023).reset_index(drop=True)
        print(len(data0), len(Corr))
    
    NC = name.split('.')[-1]
    if NC == 'N':
        a1 = sns.regplot(
            ax=ax1, data=data0, x='dist', y='corrN', 
            lowess=True,
            marker='.', color=colors_text['Normal'], 
            scatter_kws={'s': dot_size, 'color': colors_nc['Normal']},
        )
    elif NC == 'C':
        a1 = sns.regplot(
            ax=ax1, data=data0, x='dist', y='corrC', 
            lowess=True,
            marker='.', color=colors_text['Cancer'], 
            scatter_kws={'s': dot_size, 'color': colors_nc['Cancer']},
        )
    else:
        a1 = sns.regplot(
            ax=ax1, data=data0, x='dist', y='corrN', label='Normal',
            lowess=True,
            scatter_kws={'marker': '.', 's': dot_size, 'color': colors_nc['Normal']},
            line_kws={'color': colors_text['Normal'], 'label': 'Normal'},
        )
        a2 = sns.regplot(
            ax=ax1, data=data0, x='dist', y='corrC', label='Cancer',
            lowess=True,
            scatter_kws={'marker': '.', 's': dot_size, 'color': colors_nc['Cancer']},
            line_kws={'color': colors_text['Cancer'], 'label': 'Cancer'},
        )
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=[(handles[1], handles[0]), (handles[3], handles[2])], 
            labels=[labels[0], labels[2]],
            loc='upper right', bbox_to_anchor=(1, 1.04), 
            fontsize=10, scatterpoints=3,
            ncol=2, handlelength=1.2, columnspacing=0.8, handletextpad=0.5, borderpad=0.4
        )

    ax1.set_ylim(-0.72, 1.1)
    ax1.set_xlim(-cut*0.03, cut*1.05)
    ax1.tick_params(
        axis='both', labelsize=12, length=2, 
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.set_xlabel('Distance between adjacent CpG sites', fontsize=12)
    ax1.set_ylabel('Pearson correlation', fontsize=12)

    ax1.axhline(y=0.3, xmin=-cut*0.03, xmax=cut*1.05, linewidth=1, color='black', linestyle='--')
    ax1.text(cut*1.05, 0.3, '0.3', horizontalalignment='right', verticalalignment='bottom', fontsize=10)
    
    ax1.set_facecolor('None')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    fig.savefig(os.path.join(eval_root, f'{name}_cut{cut}f{frac}.svg'),
                bbox_inches='tight', pad_inches=0.01)
    # plt.show()
    data0.to_csv(os.path.join(eval_root, f'{name}_cut{cut}f{frac}.csv'), index=False)


def Dist2Corr(DistancesAll):
    Corr = []
    for ch in range(1, 23):
        info, corrN, corrC = DistancesAll[ch]['info'], DistancesAll[ch]['corr_normal'], DistancesAll[ch]['corr_cancer']
        dists = np.array([info.loc[i+1, 'MAPINFO'] - info.loc[i, 'MAPINFO'] for i in range(len(info) - 1)])
        res = np.hstack([np.array([dists]).T, np.array([corrN]).T, np.array([corrC]).T])
        Corr.append(res)
    Corr = pd.DataFrame(np.vstack(Corr), columns=['dist', 'corrN', 'corrC'])
    return Corr


def DistHist(data, name='', eval_dir=eval_root):
    
    fig = plt.figure(figsize=(3.8, 2.8), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    a1 = sns.histplot(
        ax=ax1, x=data,
        log_scale=True, stat='percent', binwidth=0.3,
        line_kws={'lw': 2},
        kde=True, linewidth=1.5,
        color='#8594B7',
    )
    
    ax1.tick_params(
        axis='both', labelsize=12, length=2, 
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.set_xlabel('Distance between adjacent CpG sites', fontsize=12)
    ax1.set_ylabel('Percentage', fontsize=12)

    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)

    fig.savefig(os.path.join(eval_dir, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()

    counts = [p.get_height() for p in ax1.patches]
    bin_centers = [p.get_x() for p in ax1.patches]
    # print(bin_centers, '\n', counts)
    hist_data = pd.DataFrame({'Bin_center': bin_centers, 'Percentage': counts})
    hist_data.to_csv(os.path.join(eval_dir, f'{name}.csv'), index=False)


def PairIdx(raw):
    pairs = pd.read_csv(os.path.join(data_root, 'group_gene', 'pair_barcode.csv'))
    pairs.columns = ['index'] + list(pairs.columns[1:])
    
    class_idx = dict()
    class_idx['pair_cancer'] = [kk for kk in pairs[pairs['label'] == 0]['index']]
    class_idx['pair_normal'] = [kk for kk in pairs[pairs['label'] == 1]['index']]
    class_idx['indi_cancer'] = [kk for kk in set(range(460)) - set(pairs[pairs['label'] == 0]['index'])]
    class_idx['indi_normal'] = [kk for kk in set(range(460, 492)) - set(pairs[pairs['label'] == 1]['index'])]
    
    assert (pairs[pairs['label'] == 0]['barcode'].values == 
            raw.columns[[kk + 3 for kk in pairs[pairs['label'] == 0]['index']]].values).all()
    assert (pairs[pairs['label'] == 1]['barcode'].values == 
            raw.columns[[kk + 3 for kk in pairs[pairs['label'] == 1]['index']]].values).all()
    assert (set(range(492)) == 
        set(class_idx['pair_cancer']) | set(class_idx['pair_normal']) | 
        set(class_idx['indi_cancer']) | set(class_idx['indi_normal']))
    return class_idx


def HandlePlotData(newX, class_idx):
    data = []
    for kk, vv in class_idx.items():
        subject, label = kk.split('_')
        dfi = pd.DataFrame({
            'Comb': kk,
            'Subject': metric_dict[subject],
            'Label': metric_dict[label],
            'x': newX[vv, 0],
            'y': newX[vv, 1],
        })
        data.append(dfi)
    data = pd.concat(data, axis=0).reset_index(drop=True)
    return data

def PValues_4group(data):
    pvalues = dict()
    PC, PN, IC, IN = [
        data[(data['Subject']=='Paired') & (data['Label']=='Cancer')],
        data[(data['Subject']=='Paired') & (data['Label']=='Normal')],
        data[(data['Subject']=='Individual') & (data['Label']=='Cancer')],
        data[(data['Subject']=='Individual') & (data['Label']=='Normal')],
    ]
    for vv in ['x', 'y']:
        pvalues[f'T_{vv}_P.CN'] = TTestPV_ind(PC[vv].values, PN[vv].values)
        pvalues[f'T_{vv}_I.CN'] = TTestPV_ind(IC[vv].values, IN[vv].values)
        pvalues[f'T_{vv}_C.PI'] = TTestPV_ind(PC[vv].values, IC[vv].values)
        pvalues[f'T_{vv}_N.PI'] = TTestPV_ind(PN[vv].values, IN[vv].values)
        pvalues[f'T_{vv}_pop'] = TTestPV_ind(
            data[data['Subject']=='Paired']['x'].values, 
            data[data['Subject']=='Individual']['x'].values
        )

    return pvalues


def PValues_2group(data):
    pvalues = dict()
    C, N, I, P = [
        data[data['Label']=='Cancer'],
        data[data['Label']=='Normal'],
        data[data['Subject']=='Individual'],
        data[data['Subject']=='Paired'],
    ]
    for vv in ['x', 'y']:
        pvalues[f'T_{vv}_CN'] = TTestPV_ind(C[vv].values, N[vv].values)
        pvalues[f'T_{vv}_PI'] = TTestPV_ind(P[vv].values, I[vv].values)
    return pvalues


def prepare_for_pca():
    raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
    raw = raw.iloc[2:, :(3 + 492)].reset_index(drop=True)
    raw[['CHR', 'MAPINFO']] = raw[['CHR', 'MAPINFO']].astype(float).astype(int)
    class_idx = PairIdx(raw)
    value_a = raw.iloc[:, 3:].values

    newX = get_pca(2055, value_a)
    plot_data = HandlePlotData(newX, class_idx)

    save = {'newX': newX, 'data': plot_data}
    with open(os.path.join(eval_root, 'raw_PCA.pkl'), 'wb') as f:
        pickle.dump(save, f)


# def get_tsne(seed, data):
#     pipeline = Pipeline([
#         ('scaling', StandardScaler()),
#         ('tsne', TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed))
#     ])
#     return pipeline.fit_transform(data.T)


def get_pca(seed, data):
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=seed))
    ])
    return pipeline.fit_transform(data.T)


def ScatterBoxHue4(data, pv='T', name=''):
    pvalues = PValues_4group(data)

    colors = {'Paired': '#74A9CF', 'Individual': '#8DD3C7'}  # 蓝，绿
    colors_nc = {'Cancer': '#E6A8A5', 'Normal': '#CBCBCB'}  # 红，灰
    colors_text = {'Paired': '#20425C', 'Individual': '#2A6E63', 'Cancer': '#99302B', 'Normal': '#676767'}
    styles = {'Cancer': '^', 'Normal': 'o'}
    data.rename(columns={'Label': 'Class'}, inplace=True)
    
    big, small = 4, 1.4
    width = 0.7
    
    fig = plt.figure(figsize=(big + small, big), facecolor='none')
    gs = fig.add_gridspec(
        1, 2,  width_ratios=(big, small),
        wspace=0.04,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_ync = fig.add_subplot(gs[0, 1], sharey=ax1)

    a1 = sns.scatterplot(
        ax=ax1, data=data, x='x', y='y', hue='Subject', style='Class',
        markers=styles, palette=colors, legend='full',
    )
    ax1.tick_params(
        axis='both', labelsize=12, length=2, 
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    )
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    # boxplots
    configs = {'width': width, 'flierprops': {'marker': '.'}}  # , 'gap': 0.1
    sns.boxplot(
        ax=ax_ync, data=data, x='Subject', y='y', hue='Class', palette=colors_nc,
        **configs, order=['Paired', 'Individual'], hue_order=['Cancer', 'Normal'], 
    )
    ax_ync.tick_params(
        axis='both', labelsize=12, length=8, 
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    )
    ax_ync.set_xlim(-width*0.75, 1+width*0.75)

    y_width = data['y'].max() - data['y'].min()
    scopey = data['y'].min() - y_width*0.05, data['y'].max() + y_width*0.05
    print(scopey)
    ax_ync.set_ylim(scopey[0], scopey[1]+20)
    ax_ync.set_xlim(-0.5, 1.5)
    ax_ync.set_ylabel('')
    ax_ync.set_xlabel('')

    ax1.legend_.remove()
    ax_ync.legend_.remove()
    
    # p-values
    ax_ync.hlines(y=scopey[0] + y_width*0.06, xmin=-width/4, xmax=width/4, linewidth=1, color=colors_text['Paired'])
    ax_ync.text(
        0, scopey[0] + y_width*0.05, '{:.2E}'.format(pvalues[f'{pv}_y_P.CN']), color=colors_text['Paired'],
        horizontalalignment='center', verticalalignment='top', fontsize=9, rotation=0
    )
    ax_ync.hlines(y=scopey[1] - y_width*0.12, xmin=-width/4, xmax=1-width/4, linewidth=1, color=colors_text['Cancer'])
    ax_ync.text(
        (1-width/2)/2, scopey[1] - y_width*0.12, '{:.2E}'.format(pvalues[f'{pv}_y_C.PI']), color=colors_text['Cancer'],
        horizontalalignment='center', verticalalignment='bottom', fontsize=9, rotation=0
    )
    ax_ync.hlines(y=scopey[1] - y_width*0.04, xmin=width/4, xmax=1+width/4, linewidth=1, color=colors_text['Normal'])
    ax_ync.text(
        (1+width/2)/2, scopey[1] - y_width*0.04, '{:.2E}'.format(pvalues[f'{pv}_y_N.PI']), color=colors_text['Normal'],
        horizontalalignment='center', verticalalignment='bottom', fontsize=9, rotation=0
    )
    for ax in [ax1, ax_ync]:
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)

    fig.savefig(os.path.join(eval_root, f'{name}.{pv}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def Dist(a, b, paired=False, dist='Euc'):
    from scipy.spatial.distance import cdist
    if paired and dist == 'Euc':
        assert a.shape == b.shape
        c = a - b
        d = np.linalg.norm(c, ord=2, axis=1)
    elif dist == 'Euc':
        c = a[:, :, np.newaxis] - b[:, :, np.newaxis].T
        d = np.linalg.norm(c, ord=2, axis=1)
        d1 = cdist(a, b, metric='euclidean')
        assert np.isclose(d, d1).all()
    elif dist == 'Cosine':
        d = 1 - cdist(a, b, metric='cosine')  # 越大越接近
        print(d[0, 1], d[1, 0], a[0]@b[1]/np.linalg.norm(a[0], ord=2)/np.linalg.norm(b[1], ord=2))
        assert np.isclose(d[0, 1], a[0]@b[1]/np.linalg.norm(a[0], ord=2)/np.linalg.norm(b[1], ord=2))
    elif dist == 'Dot':
        d = a @ b.T
    elif dist == 'L1':
        c = a[:, :, np.newaxis] - b[:, :, np.newaxis].T
        d = np.linalg.norm(c, ord=1, axis=1)
        d1 = cdist(a, b, metric='cityblock')
        assert np.isclose(d, d1).all()
    else:
        raise ValueError('{} is not supported'.format(dist))
    return d


def CalDistances(data):
    distances = dict()
    aa = data[(data['Label']=='Normal') & (data['Subject']=='Paired')][['x', 'y']].values
    bb = data[(data['Label']=='Normal') & (data['Subject']=='Individual')][['x', 'y']].values
    distances['Inner\nNormal'] = Dist(aa, bb, paired=False).flatten()
    aa = data[(data['Label']=='Cancer') & (data['Subject']=='Paired')][['x', 'y']].values
    bb = data[(data['Label']=='Cancer') & (data['Subject']=='Individual')][['x', 'y']].values
    distances['Inner\nCancer'] = Dist(aa, bb, paired=False).flatten()
    aa = data[(data['Label']=='Normal') & (data['Subject']=='Paired')][['x', 'y']].values
    bb = data[(data['Label']=='Cancer') & (data['Subject']=='Paired')][['x', 'y']].values
    distances['Inter\nPaired'] = Dist(aa, bb, paired=True).flatten()
    aa = data[(data['Label']=='Normal') & (data['Subject']=='Individual')][['x', 'y']].values
    bb = data[(data['Label']=='Cancer') & (data['Subject']=='Individual')][['x', 'y']].values
    distances['Inter\nIndividual'] = Dist(aa, bb, paired=False).flatten()
    
    data_make = []
    for kk, vv in distances.items():
        dfi = pd.DataFrame({
            'group': kk,
            'value': vv,
        })
        data_make.append(dfi)
    data_make = pd.concat(data_make, axis=0)
    return data_make, distances


def CalDistancesGlobal(data, dist='EucXY', Norm=None, simple=False):
    PC, PN, IC, IN = [
        data[(data['Subject']=='Paired') & (data['Label']=='Cancer')].index,
        data[(data['Subject']=='Paired') & (data['Label']=='Normal')].index,
        data[(data['Subject']=='Individual') & (data['Label']=='Cancer')].index,
        data[(data['Subject']=='Individual') & (data['Label']=='Normal')].index,
    ]
    # print(data.iloc[:5, :5])
    if dist == 'EucXY':
        DisGlobal = Dist(data[['x', 'y']].values, data[['x', 'y']].values, paired=False, dist='Euc')
    else:
        DisGlobal = Dist(data.iloc[:, 3:].values, data.iloc[:, 3:].values, paired=False, dist=dist)
    if Norm == 'Max':
        DisGlobal /= DisGlobal.max()
    elif Norm == 'MinMax':
        max0 = DisGlobal.max()
        min0 = DisGlobal.min()
        DisGlobal = (DisGlobal - min0) / (max0 - min0)
    
    distances = dict()
    distances['Paired-\ncancer-normal'] = DisGlobal[PN, PC].flatten()
    distances['Paired, Cancer\nPaired, Normal'] = DisGlobal[PN, :][:, PC].flatten()
    distances['Individual, Cancer\nIndividual, Normal'] = DisGlobal[IN, :][:, IC].flatten()
    if not simple:
        distances['Paired, Cancer\nIndividual, Cancer'] = DisGlobal[PC, :][:, IC].flatten()
        distances['Paired, Normal\nIndividual, Normal'] = DisGlobal[PN, :][:, IN].flatten()

    data_make = []
    for kk, vv in distances.items():
        dfi = pd.DataFrame({
            'group': kk,
            'value': vv,
        })
        data_make.append(dfi)
    data_make = pd.concat(data_make, axis=0)

    return data_make, distances


def DistanceSnsH(data, name='', ColorLabel=False):
    colors = ['#AC8580', '#F2AB7F', '#F2AB7F', '#FFD7AF', '#FFD7AF']
    colors_text = {'Paired': '#20425C', 'Individual': '#2A6E63', 'Cancer': '#99302B', 'Normal': '#676767'}
    fig = plt.figure(figsize=(2.8, 3.4), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    a1 = sns.barplot(
        ax=ax1, data=data, y='group', x='value', orient='h', 
        palette=colors, #hue='group',
        errorbar='se',  # http://seaborn.pydata.org/tutorial/error_bars.html
        # sd: +-std, se: +-std/sqrt(N), (ci, 0.95): +-1.96*se
        width=0.8,
    )

    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    
    if ColorLabel:
        ax1.tick_params(
            axis='both', labelsize=12, length=2, 
            bottom=True, top=False, left=True, right=False,
            labelbottom=True, labeltop=False, labelleft=False, labelright=False,
        )
        # Custom labels
        configs = {'horizontalalignment': 'right', 'verticalalignment': 'center', 'fontsize': 12}
        x1, x2 = -0.43, -0.08
        ax1.text(x1, 0, 'Pair', **configs)
        ax1.text(x1, 1, 'Inter', **configs)
        ax1.text(x1, 2, 'Inter', **configs)
        ax1.text(x1, 3, 'Inner', **configs)
        ax1.text(x1, 4, 'Inner', **configs)
        ax1.text(x2, 0, 'Paired', **configs, color=colors_text['Paired'])
        ax1.text(x2, 1, 'Paired', **configs, color=colors_text['Paired'])
        ax1.text(x2, 2, 'Individual', **configs, color=colors_text['Individual'])
        ax1.text(x2, 3, 'Normal', **configs, color=colors_text['Normal'])
        ax1.text(x2, 4, 'Cancer', **configs, color=colors_text['Cancer'])
    else:
        ax1.tick_params(
            axis='both', labelsize=12, length=2, 
            bottom=True, top=False, left=True, right=False,
            labelbottom=True, labeltop=False, labelleft=True, labelright=False,
        )

    ax1.set_xlabel('Distance between samples', fontsize=12)
    ax1.set_ylabel('')
    ax1.set_ylim(4.6, -0.6)

    fig.savefig(os.path.join(eval_root, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def loadX():
    Xall = np.load(os.path.join(data_root, 'group_gene', 'X_normL2.npy'))
    Yall = np.load(os.path.join(data_root, 'group_gene', 'Y.npy'))
    pair = np.load(os.path.join(data_root, 'group_gene', 'pair_matrix.npy'))
    train_idx = list(range(492))
    tr_X, tr_Y = Xall[train_idx], Yall[train_idx]
    print(tr_X.shape)
    add_ones = np.ones((tr_X.shape[0], 1))
    tr_X = np.hstack([tr_X, add_ones])
    print(tr_X.shape)
    
    class_idx = dict()
    class_idx['pair_cancer'] = np.intersect1d(np.where(pair==1)[0], np.where(tr_Y==0))
    class_idx['pair_normal'] = np.intersect1d(np.where(pair==1)[0], np.where(tr_Y==1))
    class_idx['indi_cancer'] = list(set(range(460)) - set(class_idx['pair_cancer']))
    class_idx['indi_normal'] = list(set(range(460, 492)) - set(class_idx['pair_normal']))

    return tr_X, tr_Y, class_idx


def GetEmbedding(X, model, epoch, relu=True):
    pt = torch.load(os.path.join(result_root, model, 'pth', f'epoch{epoch}.pt'))
    tt = torch.from_numpy(X).float().mm(pt['beta'].view(-1).diag()).mm(pt['emb_head.0.weight'].T) + pt['emb_head.0.bias']
    if relu:
        tt = F.relu(tt)
    return tt


def Embedding2D(value_a, class_idx, seed=None):
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        # ('tsne', TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed))
        ('pca', PCA(n_components=2, random_state=seed))
    ])
    newX = pipeline.fit_transform(value_a)

    data = []
    for kk, vv in class_idx.items():
        subject, label = kk.split('_')
        dfi = pd.DataFrame({
            'Comb': kk,
            'Subject': metric_dict[subject],
            'Label': metric_dict[label],
            'x': newX[vv, 0],
            'y': newX[vv, 1],
        })
        data.append(dfi)
    data = pd.concat(data, axis=0).reset_index(drop=True)
    return data


def prepare_for_startend(model_name):
    X, Y, class_idx = loadX()
    seed = 2047
    # start
    epoch = 0
    start = GetEmbedding(X, model_name, epoch)
    start_pca = Embedding2D(start, class_idx, seed)
    with open(os.path.join(eval_root, f'{model_name}_{epoch}_PCA.pkl'), 'wb') as f:
        pickle.dump(start_pca, f)

    # end
    epoch = 49
    end = GetEmbedding(X, model_name, epoch)
    print(start.shape, end.shape)
    end_pca = Embedding2D(end, class_idx, seed)
    with open(os.path.join(eval_root, f'{model_name}_{epoch}_PCA.pkl'), 'wb') as f:
        pickle.dump(end_pca, f)


def prepare_for_startend_cos(model_name):
    X, Y, class_idx = loadX()
    start = GetEmbedding(X, model_name, 0)  # [492, 13096]
    end = GetEmbedding(X, model_name, 49)  # [492, 13096]

    res = []
    for newX in [start.numpy(), end.numpy()]:
        data = []
        for kk, vv in class_idx.items():
            subject, label = kk.split('_')
            dfi = pd.DataFrame(newX[vv, :], index=np.arange(len(vv)))
            dfi.insert(0, 'Comb', kk)
            dfi.insert(1, 'Subject', metric_dict[subject])
            dfi.insert(2, 'Label', metric_dict[label])
            data.append(dfi)
        res.append(pd.concat(data, axis=0).reset_index(drop=True))
    return res


def StartEndPCA(start_PCA, end_PCA, name):
    colors_nc = {'Cancer': '#E6A8A5', 'Normal': '#CBCBCB'}  # red, gray

    for tt in [start_PCA, end_PCA]:
        tt = tt[tt['Subject']=='Paired']
        print('start_PCA:\tx.min=', tt['x'].min(),
              'x.max=', tt['x'].max(),
              'y.min=', tt['y'].min(),
              'y.max=', tt['y'].max())
    scopex = -72, 112
    scopey = -60, 45
    fig = plt.figure(figsize=(3.2, 6.4/(scopex[1] - scopex[0])*(scopey[1] - scopey[0])), facecolor='none')

    gs = fig.add_gridspec(
        2, 1,  height_ratios=(1, 1),
        wspace=0.05, hspace=0.05
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    
    sns.scatterplot(
        ax=ax1, data=start_PCA[start_PCA['Subject']=='Paired'], x='x', y='y', hue='Label',
        marker='o', palette=colors_nc, 
        edgecolors='face', legend='full', linewidths=0,
    )
    sns.scatterplot(
        ax=ax2, data=end_PCA[end_PCA['Subject']=='Paired'], x='x', y='y', hue='Label',
        marker='o', palette=colors_nc, 
        edgecolors='face', legend='full', linewidths=0,
    )
    
    ax1.set_xlim(*scopex)
    ax2.set_xlim(*scopex)
    ax1.set_ylim(*scopey)
    ax2.set_ylim(*scopey)

    ax1.tick_params(
        axis='both', labelsize=12, length=2, 
        bottom=True, top=False, left=True, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )
    ax2.tick_params(
        axis='both', labelsize=12, length=2, 
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles,
        ['Paired, ' + kk for kk in labels],
        # loc='best',
        loc='lower right',
        fontsize=10,
        ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
    )
    ax2.legend_.remove()
    
    ax1.text((scopex[0] + scopex[1]) / 2, 0.05*scopey[0] + 0.95*scopey[1], 'Start', horizontalalignment='center', verticalalignment='top', fontsize=10)
    ax2.text((scopex[0] + scopex[1]) / 2, 0.05*scopey[0] + 0.95*scopey[1], 'End', horizontalalignment='center', verticalalignment='top', fontsize=10)
    
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)

    fig.savefig(os.path.join(eval_root, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()
    pd_writer = pd.ExcelWriter(os.path.join(eval_root, f'{name}.xlsx'))
    start_PCA[start_PCA['Subject']=='Paired'].to_excel(pd_writer, sheet_name='paired_start', index=False)
    end_PCA[end_PCA['Subject']=='Paired'].to_excel(pd_writer, sheet_name='paired_end', index=False)
    pd_writer.save()


def DistanceStartEnd(data, name=''):
    colors = ['#F2AB7F', '#FFD7AF']

    fig = plt.figure(figsize=(2.5, 2), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    a1 = sns.barplot(
        ax=ax1, data=data, y='group', x='value', hue='Stage', orient='h',
        palette=colors, errorbar='se',
    )
    ax1.legend(
        loc='best', fontsize=10,
        ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
    )

    ax1.tick_params(
        axis='both', labelsize=12, length=2, 
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    
    ax1.set_xlabel('Distances between samples', fontsize=12)
    ax1.set_ylabel('')
    title = {
        'Cosine': 'Cosine', 'Dot': 'Inner product', 'Euc': 'Euclidean', 'L1': 'L1', 'PCABar': ''
    }
    ax1.set_title(title[name.split('End')[1]], fontsize=12, weight='bold')

    fig.savefig(os.path.join(eval_root, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def segment(df, locs, head='fea_name', dmr=None):
    if dmr:
        sco = pd.read_csv(os.path.join(result_root, 'other', model_dict_inv['DMRcate'], 'DMR_info.csv'))
        pd.testing.assert_series_equal(df[head], sco[head], check_index=False)
        df.loc[:, 'DMRcate'] = sco[dmr].values
    slc = df.iloc[locs[0]:locs[1]+1, :].copy()
    slc.loc[:, 'Dist_Sub'] = np.nan
    slc.loc[:, 'Dist_Prev'] = np.nan
    assert slc['group'].nunique() == 1
    for i, idx in enumerate(slc.index):
        slc.loc[idx, 'Dist_Sub'] = df.loc[idx+1, 'MAPINFO'] - df.loc[idx, 'MAPINFO']
        slc.loc[idx, 'Dist_Prev'] = df.loc[idx, 'MAPINFO'] - df.loc[idx-1, 'MAPINFO']
    slc.drop(['fea_ind', 'group'], axis=1, inplace=True)
    # assert slc['Dist_Sub'].values[:-1].max()<500 and slc['Dist_Prev'].values[1:].max()<500
    return slc


def UpsetPlot(eval_root0, method_list, name, top=50, unit='site'):
    from upsetplot import UpSet, from_contents, plot, query

    res = pd.read_csv(os.path.join(eval_root0, 'fea_slc_rank.csv'))
    slc_top = dict()
    if unit == 'site':
        title = f'Top {top} sites'
        for method in method_list:
            rank = res[res[method] <= top]
            print(method, len(rank))
            if len(rank) > top:
                continue
            slc_top[method] = rank['fea_name'].values
    elif unit == 'gene':
        title = f'Top {top} sites corresponding genes'
        for method in method_list:
            rank = res[res[method] <= top]['gene_set'].values
            print(method, len(set(rank)))
            if len(set(rank)) > top:
                continue
            slc_top[method] = rank
    else:
        raise ValueError(f'wrong unit: {unit}')
    pd.DataFrame(slc_top).to_csv(os.path.join(eval_root0, f'{name}_Top{top}{unit}.csv'), index=False)

    if unit in ['gene']:
        slc_top = {kk: set(vv) for kk, vv in slc_top.items()}
    data = from_contents(slc_top)
    result = query(data)
    x = result.subset_sizes.shape[0]
    y = len(slc_top)
    print(f"x: {x}, y: {y}")
    fig, ax = plt.subplots(figsize=(x*0.4 + (1 if unit in ['gene'] else 0), y*0.5), facecolor='none')

    plot_result = plot(
        data,
        fig=fig,
        sort_categories_by='-input',
        show_counts='%d',
        element_size=24,
        intersection_plot_elements=6,
        totals_plot_elements=0 if unit not in ['gene'] else 3,
    )
    plot_result["intersections"].set_ylabel("Intersection size", fontsize=12)
    if unit in ['gene']:
        plot_result["totals"].tick_params(
            axis='x',
            bottom=False, top=True, left=False, right=False,
            labelbottom=False, labeltop=True, labelleft=False, labelright=False,
        )
        # plot_result["totals"].set_xlabel("Set size", fontsize=12)
        plot_result["totals"].text(
            plot_result["totals"].get_xlim()[0] / 2, y + 1, "Set size",
            horizontalalignment='center', verticalalignment='center',
            fontsize=12,
        )
        plot_result["totals"].spines['bottom'].set_visible(False)
        plot_result["totals"].spines['top'].set_visible(True)
    # max_y = plot_result["intersections"].get_ylim()[1]
    # plot_result["intersections"].text(
    #     x / 2 - 0.5, max_y * 1.05, title,
    #     horizontalalignment='center', verticalalignment='bottom',
    #     fontsize=12, weight='bold'
    # )

    ax.tick_params(
        axis='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_title(title, fontsize=12, weight='bold')
    ax.text(0.5, 1.04, title, horizontalalignment='center', verticalalignment='bottom', fontsize=12, weight='bold')

    plt.savefig(os.path.join(eval_root0, f'{name}_top{top}{unit}.svg'), bbox_inches='tight', pad_inches=0.01)
    result.data.to_csv(os.path.join(eval_root0, f'{name}_data.csv'), index=True)


if __name__ == '__main__':
    main()
