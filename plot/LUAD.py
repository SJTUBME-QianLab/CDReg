import os
import pickle
import scipy
import pandas as pd
import numpy as np
import scipy.stats
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

from simulation import TTestPV, TTestPV_ind
import sys
sys.path.append('./../CDReg/')
from tools.dataloader import MethylationData, Dataloader

plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

os.makedirs('./LUAD', exist_ok=True)
result_root = './../results/LUAD/'
test_root = './../results/testing/LUAD/'
eval_root = './'
data_root = './../data/LUAD'
model_dict = {
    'LASSO': 'Lasso', 'Enet0.8': 'ENet', 'SGLASSO': 'SGLasso',
    'L10.5L210.2Ls1.2Lc0.3_lr0.001': 'CDReg',
    'L10.5L210.2Ls1.2Lc0_lr0.001': 'CDReg w/o C',
}
model_dict_inv = {v: k for k, v in model_dict.items()}


def main():
    settle_results()

    data = pd.read_excel(os.path.join(eval_root, 'Results_LUAD.xlsx'), sheet_name='Top25sites')
    data['logP'] = - np.log10(data['P-value'])
    DotWeightPvalue(data, name='Fig4a_Top25sites')

    model_name = model_dict_inv['CDReg']
    if not os.path.exists(f'./LUAD/{model_name}_49_TSNE.pkl'):
        prepare_for_startend(model_name)

    with open(f'./LUAD/{model_name}_0_TSNE.pkl', 'rb') as f:
        start_TSNE = pickle.load(f)
    with open(f'./LUAD/{model_name}_49_TSNE.pkl', 'rb') as f:
        end_TSNE = pickle.load(f)
    StartEndTSNE(start_TSNE, end_TSNE, 'Fig4c_TSNEStartEnd')

    dist_start, _ = CalDistancesGlobal(start_TSNE, Norm=None, simple=True)
    dist_end, _ = CalDistancesGlobal(start_TSNE, Norm=None, simple=True)
    dist_start.insert(loc=0, column='Stage', value='Start')
    dist_end.insert(loc=0, column='Stage', value='End')
    dataVS = pd.concat([dist_start, dist_end], axis=0)

    DistanceStartEnd(dataVS, name='Fig4d_TSNEBarStartEnd')

    if not os.path.exists('./LUAD/raw_use.csv'):
        prepare_for_cluster()
    raw_use = pd.read_csv('./LUAD/raw_use.csv')
    df_use = pd.read_csv('./LUAD/fea_weight.csv')
    LUADCluster(raw_use, df_use, name='Fig4e_cluster')

    fix = 24
    if not os.path.exists(f'./LUAD/folds_1site_{fix}.xlsx'):
        prepare_for_clf(fix)
    df = pd.read_excel(f'./LUAD/folds_1site_{fix}.xlsx', sheet_name=None)
    PValues = pd.read_excel(f'./LUAD/folds_1site_{fix}.xlsx', sheet_name='Ttest', index_col=0)
    PlotBar(df, 'acc', pv=PValues, name='Fig4f_BarSVMRF')
    PlotBar(df, 'f1score', pv=PValues, name='Fig4f_BarSVMRF')

    if not os.path.exists('./LUAD/AdjCorr.pkl'):
        prepare_for_corr()
    with open('./LUAD/AdjCorr.pkl', 'rb') as f:
        DistancesAll = pickle.load(f)
    ScatterDistCorr(DistancesAll, name='exFig1a_ScatterDistCorr', cut=5050, frac=0.01)

    Corr_raw = Dist2Corr(DistancesAll)
    DistHist(Corr_raw, name='exFig1b_DistHist')

    if not os.path.exists('./LUAD/raw_TSNE.pkl'):
        prepare_for_tsne()
    with open('./LUAD/raw_TSNE.pkl', 'rb') as f:
        save1 = pickle.load(f)
    data = save1['data']
    ScatterBoxHue4(data.copy(), pv='T', name='exFig1c_TSNEBox')
    dist, _ = CalDistancesGlobal(data, Norm=None, simple=False)
    DistanceSnsH(dist.copy(), name='exFig1d_TSNEBar')


def settle_results():
    # Gene of all sites and all methods
    gene_info = pd.read_csv(os.path.join(data_root, 'group_gene', 'info.csv'))
    gene_info = gene_info.drop(0, axis=0).reset_index(drop=True)
    df_use = gene_info[['IlmnID', 'gp_idx', 'CHR', 'MAPINFO', 'gene_set']]
    df_use.reset_index(drop=False, inplace=True)
    df_use.rename(columns={'index': 'fea_ind', 'IlmnID': 'fea_name', 'gp_idx': 'group'}, inplace=True)
    for mm, mmn in model_dict.items():
        if mmn == 'CDReg':
            df = pd.read_excel(os.path.join(result_root, mm, 'eval_FS.xlsx'), sheet_name='final_weight')
        elif mmn == 'CDReg w/o C':
            continue
        else:
            df = pd.read_excel(os.path.join(result_root, '3m_default100_0.0001', mm, 'eval_FS.xlsx'),
                               sheet_name='final_weight')
        df = df[['fea_ind', 'fea_name', 'ch', 'loc', 'abs_weight_normalize']]
        df.rename(columns={'ch': 'group', 'loc': 'MAPINFO', 'IlmnID':
            'fea_name', 'abs_weight_normalize': mmn},
                  inplace=True)
        df_use = pd.merge(df_use, df, on=['fea_ind', 'fea_name', 'group', 'MAPINFO'])
        assert len(df_use) == len(df)
    df_use['diff_loc'] = 0
    for i in range(len(df_use) - 1):
        if df_use.loc[i + 1, 'CHR'] != df_use.loc[i, 'CHR']:
            continue
        df_use.loc[i, 'diff_loc'] = df_use.loc[i + 1, 'MAPINFO'] - df_use.loc[i, 'MAPINFO']
    df_use_sort = df_use.sort_values(['CDReg'], ascending=False, kind='mergesort')
    df_use_sort.reset_index(drop=True, inplace=True)
    df_use_sort.insert(df_use.shape[1] - 1, 'rank', np.arange(1, 1 + len(df_use)))
    df_use_sort.to_csv('./LUAD/fea_slc.csv', index=False)

    # P-value for all sites
    pvals = pd.read_csv(os.path.join(data_root, 'part_data', 'pvalues.csv'))
    raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
    probes = raw[['IlmnID']][1:]
    probe_pvals = pd.concat([probes, pvals.iloc[1:, 1]], axis=1)
    probe_pvals.to_csv('./LUAD/probe_pvals.csv', index=False)

    # Top 25 sites with P-value and weight
    top_probes = df_use_sort[['rank', 'fea_name', 'CHR', 'MAPINFO', 'gene_set', 'CDReg']][:25]
    top_probes.rename(columns={'rank': 'Rank', 'fea_name': 'IlmnID', 'CDReg': 'Weight'}, inplace=True)
    top_probes = pd.merge(top_probes, probe_pvals, on='IlmnID')
    top_probes.rename(columns={'pvals': 'P-value'}, inplace=True)
    top_probes.to_csv('./LUAD/Top25sites.csv', index=False)

    # w OR w/o Contrast Scheme
    ours = pd.read_excel(os.path.join(result_root, model_dict_inv['CDReg'], 'eval_FS.xlsx'),
                         sheet_name='acc_allT')
    ours = ours[['fea_ind', 'fea_name']]
    ours.columns = ['ours_ind', 'ours_name']
    ours['ours_rank'] = np.arange(1, 51)
    Lcon = pd.read_excel(os.path.join(result_root, model_dict_inv['CDReg w/o C'], 'eval_FS.xlsx'),
                         sheet_name='acc_allT')
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
    concat.to_csv('./LUAD/Lcon_merge.csv', index=False)

def prepare_for_cluster():
    df = pd.read_excel(os.path.join(eval_root, 'Results_LUAD.xlsx'), sheet_name='cluster')
    df_use = df.iloc[13:19, :]

    df_use = df_use[['fea_name', 'CHR', 'MAPINFO', 'Ours']]
    probe_pvals = pd.read_csv('./LUAD/probe_pvals.csv')

    df_use = pd.merge(df_use, probe_pvals, left_on='fea_name', right_on='IlmnID').drop(['fea_name'], axis=1)
    df_use.to_csv('./LUAD/fea_weight.csv', index=False)

    raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
    raw_use = pd.merge(raw, df_use[['IlmnID']], on='IlmnID')
    raw_use = raw_use.iloc[:, :(3 + 492)]
    raw_use.loc[-1] = ['-1', -1, -1] + [0] * 460 + [1] * 32
    raw_use.index = raw_use.index + 1
    raw_use.sort_index(inplace=True)
    raw_use.to_csv('./LUAD/raw_use.csv', index=False)


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

    plt.savefig(f'./LUAD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def LUADCluster(raw_use, df_use, name):
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
    c_ave = np.mean(data_need_c.iloc[:, 3:], axis=1)
    samples_c = random.sample(list(range(3, data_need_c.shape[1])), sample_num)
    data_need_n = raw_use.iloc[1:, [0, 1, 2] + np.where(label == 1)[0].tolist()]
    n_ave = np.mean(data_need_n.iloc[:, 3:], axis=1)
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
    ins1 = ax2.plot(locs, df_use['pvals'] * 10000,
        color='seagreen', label='P-value', linewidth=1.2, marker='^', markersize=4
    )
    ax2.set_xlim(0, xlim)
    ax2.grid(axis='x', linewidth=0.8, linestyle=':')
    ax2.set_ylim(-0.7, 7)
    ax2.set_yticks(ticks=[0.0, 3.0, 6.0])
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax2.set_ylabel('P-value (1E-3)', fontsize=12)
    ax2.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    set_axis(ax2)

    # ################# ax2.twinx: weight
    ax3 = ax2.twinx()
    ins2 = ax3.plot(locs, df_use['Ours'],
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
        lns, labs, loc='lower center', fontsize=10, bbox_to_anchor=(0.36, -0.05), 
        ncol=2, handlelength=1.0, columnspacing=0.5, handletextpad=0.2, borderpad=0.5
    )
    
    # ################# x-label
    locs0 = locs.copy()
    locs0 = locs0.astype(int)
    locs0.loc[ax_idx] = ''
    ax1.set_xticks(ticks=locs)
    ax2.set_xticks(ticks=locs, labels=locs0)
#     ax2.text(0.5, -0.6, 'Location (+%d)\nCHR 7, GENE WIPI2' % offset, 
#              horizontalalignment='center', verticalalignment='bottom', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.5, -0.6, 'CHR 7, Location (+%d)\n' % offset, 
             horizontalalignment='center', verticalalignment='bottom', fontsize=12, transform=ax2.transAxes)

    plt.savefig(f'./LUAD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def get_average(folds, clf, metric):
    evals_folds = []
    for fold in folds:
        evals_all = dict()
        for mm, mmn in model_dict.items():
            if mmn == 'CDReg w/o C':
                continue
            df = pd.read_excel(os.path.join(test_root, mm, 's1_resample', f'fold{fold}', 'final_50_eval0.xlsx'),
                               sheet_name=clf)
            evals_all[mmn] = df[metric].values
        evals_all = pd.DataFrame(evals_all)  # column=50, row=methods
        evals_folds.append(evals_all)  # ten-times repeat

    folds = np.stack([dd.values for dd in evals_folds], axis=0)  # 10*50*methods
    folds_ave = pd.DataFrame(folds.mean(axis=0), columns=evals_folds[0].columns)  # 50*methods，10-time average
    folds_std = pd.DataFrame(folds.std(axis=0), columns=evals_folds[0].columns)  # 50*methods，10-time std
    return evals_folds, folds_ave, folds_std


def get_pval(folds_1site, metric_name):
    metric = metric_name.split('_')[1]
    if metric in ['acc', 'auc', 'f1score']:
        side = 'greater'
    elif metric in ['indi', 'isol']:
        side = 'less'
    else:
        raise ValueError(metric)
    table = pd.DataFrame()
    ours = folds_1site.loc[:, 'CDReg']
    for method in model_dict.values():
        if method in ['CDReg', 'CDReg w/o C']:
            continue
        pp = TTestPV(ours, folds_1site.loc[:, method], side)
        table.loc[metric_name, method] = pp
    return table


def prepare_for_clf(fix):
    folds = [88, 99, 115, 121, 150, 195, 204, 229, 246, 263]
    save_folds_1site = dict()
    save_pv_Ttest = []
    for clf in ['svm', 'rf']:
        for metric in ['acc', 'f1score']:
            evals_folds, folds_ave, _ = get_average(folds, clf, metric)  # 指定分类器、指定指标
            folds_1site = pd.DataFrame(np.stack([dd.iloc[fix, :].values for dd in evals_folds], axis=0),
                                       columns=evals_folds[0].columns)
            metric_name = '_'.join([clf, metric])
            save_folds_1site[metric_name] = folds_1site
            save_pv_Ttest.append(get_pval(folds_1site, metric_name))  # 1*方法数，df格式，index=metric_name

    df_pv_Ttest = pd.concat(save_pv_Ttest, axis=0)

    pd_writer = pd.ExcelWriter(f'./LUAD/folds_1site_{fix}.xlsx')
    for name, df in save_folds_1site.items():
        df.to_excel(pd_writer, index=False, index_label=True, sheet_name=name)
    df_pv_Ttest.to_excel(pd_writer, index=True, index_label=True, sheet_name='Ttest')
    pd_writer.save()


def PlotBar(df, metric, pv=None, name=''):
    def SnsConcatMetric(mt):
        data = []
        for clf in ['svm', 'rf']:
            dfi = df[f'{clf}_{mt}']
            for mm in ['Lasso', 'ENet', 'SGLasso', 'CDReg']:
                data.append(pd.DataFrame({
                    'idx': range(10),
                    'clf': clf,
                    'method': mm,
                    mt: dfi[mm],
                }))
        data = pd.concat(data, axis=0)
        return data

    data = pd.merge(SnsConcatMetric('acc'), SnsConcatMetric('f1score'))

    name_dict = {'svm': 'Support Vector Machines', 'lr': 'Logistic Regession', 'rf': 'Random Forest', 
                 'acc': 'Accuracy', 'f1score': 'F1-score'}
    method_list = ['Lasso', 'ENet', 'SGLasso', 'CDReg']
    colors0 = sns.color_palette('Pastel1')
    colors = colors0[1:4] + [colors0[0]]

    if metric == 'acc':
        fig = plt.figure(figsize=(10*0.22, 2.9))
        x0 = 0.99
        padding = 40
    elif metric == 'f1score':
        fig = plt.figure(figsize=(3*0.72, 2.9))
        x0 = 0.72
        padding = 36
    ax1 = plt.subplot(1, 1, 1)

    bar1 = sns.barplot(
        ax=ax1, data=data, y='clf', x=metric, orient='h', 
        order=['svm', 'rf'],
        hue='method', hue_order=method_list, palette=colors,
    )

    if metric == 'acc':
        ax1.set_xlim(0.8, 1.02)
        ax1.set_xticks([0.8, 0.9, 1.0])
    elif metric == 'f1score':
        ax1.set_xlim(0.2, 0.92)
        ax1.set_xticks([0.2, 0.5, 0.8])
    ax1.set_xlabel('')

    ax1.set_ylim(1.5, -0.5)
    if metric == 'acc':
        ax1.set_yticks(ticks=[0, 1], labels=['SVM', 'RF'], 
                       rotation=90, verticalalignment='center')
    else:
        ax1.set_yticks(ticks=[0, 1], labels=[])
    ax1.set_ylabel('')

    ax1.set_title(name_dict[metric], size=12, weight='bold')

    ax1.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )

    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_visible(False)
    
    ax1.set_facecolor('None')  # 透明背景
    for i in bar1.containers:
        bar1.bar_label(i, fmt='%.3f', padding=-padding, fontsize=10)

    if pv is not None:
        start = ax1.patches[0].get_y()
        width = ax1.patches[2].get_y() - ax1.patches[0].get_y()
        ax1.text(x0, start - width * 0.5, 'P-value',
                 horizontalalignment='center', verticalalignment='center')
        ax1.text(x0, start - width * 0.5 + 1, 'P-value',
                 horizontalalignment='center', verticalalignment='center')
        ax1.text(x0, start + 0.02, '-'*16,
                 horizontalalignment='center', verticalalignment='bottom')
        ax1.text(x0, start + 1.02, '-'*16,
                 horizontalalignment='center', verticalalignment='bottom')
        for i, mm in enumerate(method_list[:-1]):
            ax1.text(x0, start + width * (i+0.5), 
                     '{:.2e}'.format(pv.loc[f'svm_{metric}', mm]) if i!=4 else '',
                    horizontalalignment='center', verticalalignment='center')
            ax1.text(x0, start + width * (i+0.5) + 1, 
                     '{:.2e}'.format(pv.loc[f'rf_{metric}', mm]) if i!=4 else '',
                    horizontalalignment='center', verticalalignment='center')

    if metric == 'acc':
        ax1.legend(
            loc='center right', fontsize=10, bbox_to_anchor=(-0.15, 0.5), # borderaxespad=0.2, 
            ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
        )
    else:
        ax1.legend_.remove()
    SaveName = '{}_{}_{}'.format(name, metric, '' if pv is None else 'p')
    
    plt.savefig(f'./LUAD/{SaveName}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


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

    with open('./LUAD/AdjCorr.pkl', 'wb') as f:
        pickle.dump(DistancesAll, f)


def ScatterDistCorr(DistancesAll, name='', cut=0, frac=1):
    Corr = []
    for ch in range(1, 23):
        info, corrN, corrC = DistancesAll[ch]['info'], DistancesAll[ch]['corr_normal'], DistancesAll[ch]['corr_cancer']
        dists = np.array([info.loc[i + 1, 'MAPINFO'] - info.loc[i, 'MAPINFO'] for i in range(len(info) - 1)])
        res = np.hstack([np.array([dists]).T, np.array([corrN]).T, np.array([corrC]).T])
        Corr.append(res)
    Corr = pd.DataFrame(np.vstack(Corr), columns=['dist', 'corrN', 'corrC'])

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

    plt.savefig(f'./LUAD/{name}_cut{cut}f{frac}.svg',
                bbox_inches='tight', pad_inches=0.01)
    plt.show()


def Dist2Corr(DistancesAll):
    Corr = []
    for ch in range(1, 23):
        info, corrN, corrC = DistancesAll[ch]['info'], DistancesAll[ch]['corr_normal'], DistancesAll[ch]['corr_cancer']
        dists = np.array([info.loc[i+1, 'MAPINFO'] - info.loc[i, 'MAPINFO'] for i in range(len(info) - 1)])
        res = np.hstack([np.array([dists]).T, np.array([corrN]).T, np.array([corrC]).T])
        Corr.append(res)
    Corr = pd.DataFrame(np.vstack(Corr), columns=['dist', 'corrN', 'corrC'])
    return Corr



def DistHist(data, name=''):
    
    fig = plt.figure(figsize=(3.8, 2.8), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    a1 = sns.histplot(
        ax=ax1, x=data['dist'], 
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

    plt.savefig(f'./LUAD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


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
    name_dict = {
        'pair': 'Paired', 'indi': 'Individual', 
        'cancer': 'Cancer', 'normal': 'Normal',
    }
    data = []
    for kk, vv in class_idx.items():
        subject, label = kk.split('_')
        dfi = pd.DataFrame({
            'Comb': kk,
            'Subject': name_dict[subject],
            'Label': name_dict[label],
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


def prepare_for_tsne():
    raw = pd.read_csv(os.path.join(data_root, 'part_data', 'all_dropna_chr_loc.csv'))
    raw = raw.iloc[2:, :(3 + 492)].reset_index(drop=True)
    raw[['CHR', 'MAPINFO']] = raw[['CHR', 'MAPINFO']].astype(float).astype(int)
    class_idx = PairIdx(raw)
    value_a = raw.iloc[:, 3:].values

    newX = get_tsne(2055, value_a)
    plot_data = HandlePlotData(newX, class_idx)

    save = {'newX': newX, 'data': plot_data}
    with open('./LUAD/raw_TSNE.pkl', 'wb') as f:
        pickle.dump(save, f)


def get_tsne(seed, data):
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('tsne', TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed))
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
        wspace=0.06, hspace=0.06
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
    ax_ync.hlines(y=scopey[0] + y_width*0.15, xmin=-width/4, xmax=width/4, linewidth=1, color=colors_text['Paired'])
    ax_ync.text(
        0, scopey[0] + y_width*0.14, '{:.2E}'.format(pvalues[f'{pv}_y_P.CN']), color=colors_text['Paired'],
        horizontalalignment='center', verticalalignment='top', fontsize=9, rotation=0
    )
    ax_ync.hlines(y=scopey[1] - y_width*0.2, xmin=-width/4, xmax=1-width/4, linewidth=1, color=colors_text['Cancer'])
    ax_ync.text(
        (1-width/2)/2, scopey[1] - y_width*0.2, '{:.2E}'.format(pvalues[f'{pv}_y_C.PI']), color=colors_text['Cancer'], 
        horizontalalignment='center', verticalalignment='bottom', fontsize=9, rotation=0
    )
    ax_ync.hlines(y=scopey[1] - y_width*0.1, xmin=width/4, xmax=1+width/4, linewidth=1, color=colors_text['Normal'])
    ax_ync.text(
        (1+width/2)/2, scopey[1] - y_width*0.1, '{:.2E}'.format(pvalues[f'{pv}_y_N.PI']), color=colors_text['Normal'], 
        horizontalalignment='center', verticalalignment='bottom', fontsize=9, rotation=0
    )
    for ax in [ax1, ax_ync]:
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)

    plt.savefig(f'./LUAD/{name}.{pv}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def DistEuc(a, b, paired=False):
    if paired:
        assert a.shape == b.shape
        c = a - b
        d = np.linalg.norm(c, ord=2, axis=1)
    else:
        c = a[:, :, np.newaxis] - b[:, :, np.newaxis].T
        d = np.linalg.norm(c, ord=2, axis=1)
        # from scipy.spatial.distance import cdist
        # print(np.max(np.abs(d - cdist(a, b, metric='euclidean'))))
        # assert np.max(np.abs(d - cdist(a, b, metric='euclidean'))) < 1e-5
    return d


def CalDistances(data):
    distances = dict()
    aa = data[(data['Label']=='Normal') & (data['Subject']=='Paired')][['x', 'y']].values
    bb = data[(data['Label']=='Normal') & (data['Subject']=='Individual')][['x', 'y']].values
    distances['Inner\nNormal'] = DistEuc(aa, bb, paired=False).flatten()
    aa = data[(data['Label']=='Cancer') & (data['Subject']=='Paired')][['x', 'y']].values
    bb = data[(data['Label']=='Cancer') & (data['Subject']=='Individual')][['x', 'y']].values
    distances['Inner\nCancer'] = DistEuc(aa, bb, paired=False).flatten()
    aa = data[(data['Label']=='Normal') & (data['Subject']=='Paired')][['x', 'y']].values
    bb = data[(data['Label']=='Cancer') & (data['Subject']=='Paired')][['x', 'y']].values
    distances['Inter\nPaired'] = DistEuc(aa, bb, paired=True).flatten()
    aa = data[(data['Label']=='Normal') & (data['Subject']=='Individual')][['x', 'y']].values
    bb = data[(data['Label']=='Cancer') & (data['Subject']=='Individual')][['x', 'y']].values
    distances['Inter\nIndividual'] = DistEuc(aa, bb, paired=False).flatten()
    
    data_make = []
    for kk, vv in distances.items():
        dfi = pd.DataFrame({
            'group': kk,
            'value': vv,
        })
        data_make.append(dfi)
    data_make = pd.concat(data_make, axis=0)
    return data_make, distances


def CalDistancesGlobal(data, Norm=None, simple=False):
    PC, PN, IC, IN = [
        data[(data['Subject']=='Paired') & (data['Label']=='Cancer')].index,
        data[(data['Subject']=='Paired') & (data['Label']=='Normal')].index,
        data[(data['Subject']=='Individual') & (data['Label']=='Cancer')].index,
        data[(data['Subject']=='Individual') & (data['Label']=='Normal')].index,
    ]
    DisGlobal = DistEuc(data[['x', 'y']].values, data[['x', 'y']].values, paired=False)
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


def DistanceSnsH(data0, name='', ColorLabel=False):
    colors = ['#AC8580', '#F2AB7F', '#F2AB7F', '#FFD7AF', '#FFD7AF']
    colors_text = {'Paired': '#20425C', 'Individual': '#2A6E63', 'Cancer': '#99302B', 'Normal': '#676767'}
    fig = plt.figure(figsize=(2.8, 3.4), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)
    
    data = data0.copy()
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

    plt.savefig(f'./LUAD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def loadX():
    data_name = 'LUAD'
    device = torch.device('cpu')
    data = MethylationData(os.path.join(data_root, 'group_gene'), data_name, device)
    loader = Dataloader(data)
    pair = np.load(os.path.join(data_root, 'group_gene', 'pair_matrix.npy'))
    
    tr_X, tr_Y, _, _ = loader.get_whole()
    print(tr_X.shape)
    add_ones = torch.ones((tr_X.shape[0], 1)).to(tr_X.device)
    tr_X = torch.cat([tr_X, add_ones], dim=1)
    print(tr_X.shape)
    
    class_idx = dict()
    class_idx['pair_cancer'] = np.intersect1d(np.where(pair==1)[0], np.where(tr_Y==0))
    class_idx['pair_normal'] = np.intersect1d(np.where(pair==1)[0], np.where(tr_Y==1))
    class_idx['indi_cancer'] = list(set(range(460)) - set(class_idx['pair_cancer']))
    class_idx['indi_normal'] = list(set(range(460, 492)) - set(class_idx['pair_normal']))

    return tr_X, tr_Y, class_idx


def GetEmbedding(X, model, epoch, relu=True):
    result_dir = '/home/data/tangxl/ContrastSGL/results_appli/LUAD'
    pt = torch.load(os.path.join(result_dir, model, 'pth', f'epoch{epoch}.pt'))
    tt = X.mm(pt['beta'].view(-1).diag()).mm(pt['emb_head.0.weight'].T) + pt['emb_head.0.bias']
    if relu:
        tt = F.relu(tt)
    return tt


def Embedding2D(value_a, class_idx, seed=None):
    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('tsne', TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed))
    ])
    newX = pipeline.fit_transform(value_a)  # 降维后的数据

    name_dict = {
        'pair': 'Paired', 'indi': 'Individual', 
        'cancer': 'Cancer', 'normal': 'Normal',
    }
    data = []
    for kk, vv in class_idx.items():
        subject, label = kk.split('_')
        dfi = pd.DataFrame({
            'Comb': kk,
            'Subject': name_dict[subject],
            'Label': name_dict[label],
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
    with open(f'./LUAD/{model_name}_{epoch}_TSNE.pkl', 'wb') as f:
        pickle.dump(start_pca, f)

    # end
    epoch = 49
    end = GetEmbedding(X, model_name, epoch)
    print(start.shape, end.shape)
    end_pca = Embedding2D(end, class_idx, seed)
    with open(f'./LUAD/{model_name}_{epoch}_TSNE.pkl', 'wb') as f:
        pickle.dump(end_pca, f)


def StartEndTSNE(start_TSNE, end_TSNE, name):
    colors_nc = {'Cancer': '#E6A8A5', 'Normal': '#CBCBCB'}  # red, gray

    scopex = -21, 13
    scopey = -19, 4
    fig = plt.figure(figsize=(3, 6/(scopex[1] - scopex[0])*(scopey[1] - scopey[0])), facecolor='none')

    gs = fig.add_gridspec(
        2, 1,  height_ratios=(1, 1),
        wspace=0.05, hspace=0.05
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    
    sns.scatterplot(
        ax=ax1, data=start_TSNE[start_TSNE['Subject']=='Paired'], x='x', y='y', hue='Label',
        marker='o', palette=colors_nc, 
        edgecolors='face', legend='full', linewidths=0,
    )
    sns.scatterplot(
        ax=ax2, data=end_TSNE[end_TSNE['Subject']=='Paired'], x='x', y='y', hue='Label',
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
        loc='best',
        fontsize=10,
        ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
    )
    ax2.legend_.remove()
    
    ax1.text((scopex[0] + scopex[1]) / 2, scopey[1]*0.8, 'Start', horizontalalignment='center', verticalalignment='top', fontsize=10)
    ax2.text((scopex[0] + scopex[1]) / 2, scopey[1]*0.8, 'End', horizontalalignment='center', verticalalignment='top', fontsize=10)
    
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)

    plt.savefig(f'./LUAD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def DistanceStartEnd(data0, name=''):
    colors = ['#F2AB7F', '#FFD7AF']

    fig = plt.figure(figsize=(2.5, 2), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)
    
    data = data0.copy()
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

    plt.savefig(f'./LUAD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    main()
