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

from LUAD import get_pval, get_average, get_res_df, UpsetPlot, plot_method_legend

plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

eval_root = './AD/'
os.makedirs(eval_root, exist_ok=True)
# data_root = './../data/AD/'
# result_root = './../results/AD/'
# test_root = './../results/testing/AD'
data_root = '/home/data/tangxl/ContrastSGL/casecontrol_data/AD'
result_root = '/home/data/tangxl/ContrastSGL/results_appli/AD/'
test_root = '/home/data/tangxl/ContrastSGL/results_appli/0811_add/results_AD/'

model_dict = {
    'LASSO': 'Lasso', 'Enet0.8': 'ENet', 'SGLASSO': 'SGLasso', 'pclogit': 'Pclogit',
    'dmp10': 'DMP', 'DMRcate': 'DMRcate',
    # 'L10.150_L210.040_Lg0.5_Lcon1.2_lr1.00': 'CDReg',
    'L10.15L210.04Ls0.5Lc1.2_lr0.001': 'CDReg',
}
model_dict_inv = {v: k for k, v in model_dict.items()}
colorsPal = sns.color_palette('Pastel1')
colors_dict = dict(zip(
    list(model_dict.values()),
    colorsPal[1:5] + ['#EBECA6', '#FFE9D2'] + [colorsPal[0]]
))
metric_dict = {
    'acc': 'Accuracy', 'auc': 'AUC', 'f1score': 'F1-score', 'AUPRC': 'AUPRC',
    'svm': 'Support Vector Machine', 'lr': 'Logistic Regession', 'rf': 'Random Forest',
}

def main():
    settle_results()

    top = 50
    UpsetPlot(eval_root, model_dict.values(), 'Fig5a_Upset', top=top, unit='site')

    data = pd.read_csv(os.path.join(eval_root, 'Top15sitesVS.csv'))
    data = data[['IlmnID', 'CHR', 'MAPINFO', 'gene_set', 'CDReg', 'SGLasso', 'CDReg_rank', 'SGLasso_rank', 'AD-NC-adj']]
    data['logP'] = -np.log10(data['AD-NC-adj'])
    DotWeightVS2(data.copy(), name='Fig5b_Top15sites_Weight', Line=False, reverse=False)
    DotPV(data.copy(), name='Fig5b_Top15sites_PValue')

    if not os.path.exists(os.path.join(eval_root, 'folds_ave.xlsx')):
        prepare_for_clf_curve()
    ave = pd.read_excel(os.path.join(eval_root, 'folds_ave.xlsx'), sheet_name='svm_acc_ave')
    std = pd.read_excel(os.path.join(eval_root, 'folds_ave.xlsx'), sheet_name='svm_acc_std')
    CLFCurve(ave, std, 'Fig5c_svm_acc', smooth=True, fill=True, dot=False)
    ClfBarStd(std, 'Fig5d_svm_acc_BarStd')


def prepare_for_clf_curve():
    folds = [13, 39, 214, 313, 337, 346, 398, 400, 443, 444]
    save_folds_ave = dict()
    for clf in ['svm', 'rf']:
        for metric in ['acc', 'f1score']:
            evals_folds, folds_ave, folds_std = get_average(clf, metric, model_dict, folds, test_root)
            # evals_folds：a list of 10-repetition, each is shaped as 50*methods
            # folds_ave：50*methods，10-time average
            metric_name = '_'.join([clf, metric])
            save_folds_ave[metric_name+'_ave'] = folds_ave  # 50*methods
            save_folds_ave[metric_name+'_std'] = folds_std  # 50*methods
    pd_writer = pd.ExcelWriter(os.path.join(eval_root, 'folds_ave.xlsx'))
    for name, df in save_folds_ave.items():
        df.to_excel(pd_writer, index=False, index_label=True, sheet_name=name)
    pd_writer.save()


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
    probe_pvals = pd.read_csv(os.path.join(data_root, 'pval_data', 'probe_pvals.csv'))

    top_probes = df_use[['fea_name', 'CHR', 'MAPINFO', 'gene_set', 'CDReg', 'SGLasso']]
    top_probes = pd.merge(top_probes, df_rank[['fea_name', 'CDReg', 'SGLasso']].rename(columns={'CDReg': 'CDReg_rank', 'SGLasso': 'SGLasso_rank'}), on='fea_name')
    top_probes = top_probes[(top_probes['CDReg_rank'] <= 15) | (top_probes['SGLasso_rank'] <= 15)]
    top_probes.rename(columns={'fea_name': 'IlmnID'}, inplace=True)
    top_probes = pd.merge(top_probes, probe_pvals, on='IlmnID', how='left')
    top_probes.to_csv(os.path.join(eval_root, 'Top15sitesVS.csv'), index=False)


def DotWeightVS2(data, name='', Line=False, reverse=False):
    import matplotlib.patches as patches

    colors_dark = {'SGLasso': '#9161AB', 'CDReg': '#D46654'}
    colors_light = {'SGLasso': '#decbe4', 'CDReg': '#fbb4ae'}

    fig = plt.figure(figsize=(10/2.54, 21/2.54), facecolor='none')
    gs = fig.add_gridspec(
        1, 2,  width_ratios=(1, 1), wspace=0
    )
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1], sharey=ax2)
    margin = 0.18+0.18  # 0.18

    a2 = sns.stripplot(
        ax=ax2, data=data, orient='h', x='SGLasso', y='IlmnID', 
        color=colors_dark['SGLasso'], size=8, #alpha=0.6,
    )
    a3 = sns.stripplot(
        ax=ax3, data=data, orient='h', x='CDReg', y='IlmnID', 
        color=colors_dark['CDReg'], size=8, #alpha=0.6,
    )
    if Line:
        ax2.hlines(
            y=np.arange(len(data)), xmin=-margin, xmax=data['SGLasso'], 
            color='grey',  # color=colors_dark['SGLasso'], 
            linewidth=1, 
        )
        ax3.hlines(
            y=np.arange(len(data)), xmin=-margin, xmax=data['CDReg'], 
            color='grey',  # color=colors_dark['CDReg'], 
            linewidth=1, 
        )
    
    for ax in [ax2, ax3]:
        ax.set_ylim(len(data)-0.5, -0.5)
        ax.set_xlim(-margin, 1 + margin-0.18)
        ax.set_xticks([0., 0.5, 1.0])
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

    if reverse:
        ax2.invert_xaxis()

    intv = 0.06
    for i in range(len(data)):
        if data.loc[i, 'SGLasso'] > 0:
            rank = data.loc[i, 'SGLasso_rank']
            if rank <= 15:
                ax2.text(
                    data.loc[i, 'SGLasso']+intv, i, rank,
                    horizontalalignment=('right' if reverse else 'left'), verticalalignment='center', fontsize=9
                )
            else:
                ax2.text(
                    data.loc[i, 'SGLasso']-intv, i, rank,
                    horizontalalignment=('left' if reverse else 'right'), verticalalignment='center', fontsize=9
                )
        if data.loc[i, 'CDReg'] > 0:
            rank = data.loc[i, 'CDReg_rank']
            if rank <= 15:
                ax3.text(
                    data.loc[i, 'CDReg']+intv, i, rank,
                    horizontalalignment='left', verticalalignment='center', fontsize=9
                )
            else:
                ax3.text(
                    data.loc[i, 'CDReg']-intv, i, rank,
                    horizontalalignment='right', verticalalignment='center', fontsize=9
                )

    # background color
    cut = data.loc[int(np.where(data['SGLasso_rank']==15)[0]), 'SGLasso']
    bbox = patches.Rectangle(
        (cut, -1), 1 + margin - cut, len(data)+1, 
        linewidth=1, edgecolor=None, 
        facecolor=colors_light['SGLasso'], alpha=0.7 
    )
    ax2.add_patch(bbox)

    cut = data.loc[int(np.where(data['CDReg_rank']==15)[0]), 'CDReg']
    bbox = patches.Rectangle(
        (cut, -1), 1 + margin - cut, len(data)+1, 
        linewidth=1, edgecolor=None, 
        facecolor=colors_light['CDReg'], alpha=0.5 
    )
    ax3.add_patch(bbox)

    ax3.text(-margin, -1.8, 'Weight', horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    save_name = '{}.{}.{}'.format(name, 'line' if Line else '', 'rev' if reverse else '')
    fig.savefig(os.path.join(eval_root, f'{save_name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def DotPV(data, name=''):
    fig = plt.figure(figsize=(4/2.54, 21/2.54), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    a1 = sns.stripplot(
        ax=ax1, data=data, orient='h', x='logP', y='IlmnID',
        legend=False, edgecolor=None, color='grey',
    )
    print(data['logP'].max(), data['logP'].min())
    ax1.set_xlim(0.1, 3.9)
    ax1.set_xticks([1, 2, 3])
    ax1.tick_params(
        axis='both', labelsize=10, length=2, pad=0,
        bottom=False, top=True, left=False, right=False,
        labelbottom=False, labeltop=True, labelleft=False, labelright=False,
    )
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel('')
    ax1.grid(axis='y', linestyle=':')
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)

    ax1.set_xlabel('-log(P)', fontsize=12)

    fig.savefig(os.path.join(eval_root, f'{name}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def CLFCurve(df_mean, df_std, name, smooth=False, fill=False, dot=False):
    top_num = 15
    intv = 1
    top_idx = np.arange(intv, top_num + 1, intv)

    colors = {
        'Lasso': '#3376A8', 'ENet': '#32955C', 'SGLasso': '#7B4F94', 'Pclogit': '#FF9933',
        'DMP': '#C2BF41', 'DMRcate': '#BB8E1D',
        'CDReg': '#CC4A33'
    }

    fig = plt.figure(figsize=(3.2, 3.2))
    ax = plt.subplot(1, 1, 1)

    for kk, color in colors.items():
        print(kk)
        y = df_mean.loc[top_idx - 1, kk]
        std = df_std.loc[top_idx - 1, kk] * 0.5
        if smooth:
            # https://blog.csdn.net/m0_48300767/article/details/130075597
            from scipy.interpolate import make_interp_spline
            m = make_interp_spline(top_idx, y)
            xs = np.linspace(1, top_num, 500)
            ys = m(xs)
            ax.plot(
                xs, ys,
                label=kk,
                linestyle='-', linewidth=1.5, color=color, zorder=1
            )
            if dot:
                ax.scatter(
                    top_idx, y,
                    color=color, s=4, zorder=2
                )
            if fill:
                m1 = make_interp_spline(top_idx, y - std)
                ys1 = m1(xs)
                m2 = make_interp_spline(top_idx, y + std)
                ys2 = m2(xs)
                ax.fill_between(
                    xs, ys1, ys2,
                    facecolor=color, edgecolor=None, alpha=0.2
                )

        else:
            if dot:
                ax.plot(
                    top_idx, y, label=kk,
                    linestyle='-', linewidth=1.5, color=color, marker='.', markersize=4
                )
            else:
                ax.plot(
                    top_idx, y, label=kk,
                    linestyle='-', linewidth=1.5, color=color,
                )
            if fill:
                ax.fill_between(
                    top_idx, y - std, y + std,
                    facecolor=color, edgecolor=None, alpha=0.2
                )

    ax.set_xlim(0.5, top_num + 0.5)
    ax.set_xticks(ticks=np.linspace(start=1, stop=top_num, num=int(top_num/2)+1, endpoint=True))
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylim(0.605, 0.76)
    ax.set_yticks(ticks=[0.6, 0.64, 0.68, 0.72, 0.76])
    ax.set_ylabel(metric_dict[name.split('_')[2]], fontsize=12)
    # ax.set_title(metric_dict[name.split('_')[1]], fontsize=12, weight='bold')

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    plot_method_legend(colors, os.path.join(eval_root, f'{name}_legend_rows.svg'))
    plot_method_legend(colors, os.path.join(eval_root, f'{name}_legend_cols.svg'), cols=-1)
    plot_method_legend(colors, os.path.join(eval_root, f'{name}_legend_cols2.svg'), cols=2)
    # plt.legend(
    #     loc="lower left", fontsize=10, bbox_to_anchor=(1, 0), # borderaxespad=0.2,
    #     ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
    # )

    SaveName = '{}_{:d}.{:d}_{}{}{}'.format(
        name, top_num, intv, 's' if smooth else '', 'f' if fill else '', 'd' if dot else '')
    fig.savefig(os.path.join(eval_root, f'{SaveName}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def ClfBarStd(df, name=''):
    top_num = 15
    intv = 1
    top_idx = np.arange(intv, top_num + 1, intv)

    data = pd.DataFrame(
        [[kk, df.loc[top_idx - 1, kk].sum(axis=0)] for kk in model_dict.values()],
        columns=['method', 'std']
    )

    fig = plt.figure(figsize=(2.4, 3.2))
    ax1 = plt.subplot(1, 1, 1)

    bar1 = sns.barplot(
        ax=ax1, data=data, x='std', y='method', orient='h',
        palette=colors_dict,
        width=0.8,
    )

    ax1.xaxis.set_label_text('Cumulative standard deviation', fontsize=12, x=-0.2, horizontalalignment='left')
    ax1.set_ylabel('')
    ax1.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )

    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_visible(False)

    ax1.set_facecolor('None')
    for i in bar1.containers:
        bar1.bar_label(i, fmt='%.3f', padding=-40, fontsize=10)

    SaveName = '{}_{:d}.{:d}'.format(name, top_num, intv)
    fig.savefig(os.path.join(eval_root, f'{SaveName}.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


if __name__ == '__main__':
    main()

