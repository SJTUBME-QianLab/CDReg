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

from LUAD import get_pval

plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


os.makedirs('./AD', exist_ok=True)
result_root = './../results/AD/'
test_root = './../results/testing/AD/'
eval_root = './'
data_root = './../data/AD'
model_dict = {
    'LASSO': 'Lasso', 'Enet0.8': 'ENet', 'SGLASSO': 'SGLasso', 'pclogit': 'Pclogit',
    'L10.15L210.04Ls0.5Lc1.2_lr0.001': 'CDReg',
}
model_dict_inv = {v: k for k, v in model_dict.items()}


def main():
    settle_results()

    data = pd.read_excel(os.path.join(eval_root, 'Results_AD.xlsx'), sheet_name='Top15sites')
    data.rename(columns={'Unnamed: 6': 'SGLasso_rank', 'Ours': 'CDReg', 'Unnamed: 8': 'CDReg_rank'}, inplace=True)
    data = data.drop(0, axis=0).reset_index(drop=True)
    data['logP'] = -np.log10(data['P-value'])

    DotWeightVS2(data, name='Fig5a_Top15sites_VS', Line=False, reverse=False)
    DotPV(data, name='Fig5a_Top15sites_PValue')

    if not os.path.exists('./AD/folds_ave.xlsx'):
        prepare_for_clf()
    ave = pd.read_excel('./AD/folds_ave.xlsx', sheet_name='svm_acc_ave')
    std = pd.read_excel('./AD/folds_ave.xlsx', sheet_name='svm_acc_std')
    plot_line1('Fig5c_svm_acc', ave, std, smooth=True, fill=True, dot=False)


def get_average(folds, clf, metric):
    evals_folds = []
    for fold in folds:
        evals_all = dict()
        for mm, mmn in model_dict.items():
            df = pd.read_excel(os.path.join(test_root, mm, 's1_resample', f'fold{fold}', 'final_50_eval0.xlsx'),
                               sheet_name=clf)
            evals_all[mmn] = df[metric].values
        evals_all = pd.DataFrame(evals_all)  # column=50, row=methods
        evals_folds.append(evals_all)  # ten-times repeat

    folds = np.stack([dd.values for dd in evals_folds], axis=0)  # 10*50*methods
    folds_ave = pd.DataFrame(folds.mean(axis=0), columns=evals_folds[0].columns)  # 50*methods，10-time average
    folds_std = pd.DataFrame(folds.std(axis=0), columns=evals_folds[0].columns)  # 50*methods，10-time std
    return evals_folds, folds_ave, folds_std


def prepare_for_clf():
    folds = [13, 39, 214, 313, 337, 346, 398, 400, 443, 444]
    save_folds_ave = dict()
    for clf in ['svm', 'rf']:
        for metric in ['acc', 'f1score']:
            evals_folds, folds_ave, folds_std = get_average(folds, clf, metric)  # 指定分类器、指定指标
            # evals_folds：a list of 10-repetition, each is shaped as 50*methods
            # folds_ave：50*methods，10-time average
            metric_name = '_'.join([clf, metric])
            save_folds_ave[metric_name+'_ave'] = folds_ave  # 50*methods
            save_folds_ave[metric_name+'_std'] = folds_std  # 50*methods
    pd_writer = pd.ExcelWriter('./AD/folds_ave.xlsx')
    for name, df in save_folds_ave.items():
        df.to_excel(pd_writer, index=False, index_label=True, sheet_name=name)
    pd_writer.save()


def plot_line1(name, df_mean, df_std, smooth=False, fill=False, dot=False):
    top_num = 15
    intv = 1
    top_idx = np.arange(intv, top_num + 1, intv)
    name_dict = {'svm': 'Support Vector Machines', 'lr': 'Logistic Regession', 'rf': 'Random Forest', 
                 'acc': 'Accuracy', 'f1score': 'F1-score'}

    colors = ['#3376A8', '#32955C', '#7B4F94', '#E0BF3D', '#CC4A33']
    
    fig = plt.figure(figsize=(3.2, 3.2))
    ax = plt.subplot(1, 1, 1)
    
    for kk, color in zip(df_mean.columns, colors):
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
                m1 = make_interp_spline(top_idx, y-std)
                ys1 = m1(xs)
                m2 = make_interp_spline(top_idx, y+std)
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

    ax.set_xlim(0, top_num + 1)
    ax.set_xticks(ticks=np.linspace(start=0, stop=top_num, num=int(top_num/5+1), endpoint=True))
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylim(0.55, 0.75)
    ax.set_yticks(ticks=[0.56, 0.6, 0.64, 0.68, 0.72])
    ax.set_ylabel(name_dict[name.split('_')[2]], fontsize=12)
    ax.set_title(name_dict[name.split('_')[1]], fontsize=12, weight='bold')

    ax.tick_params(
        axis='both', labelsize=12, length=2, width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    plt.legend(
        loc="lower right", fontsize=10, # borderaxespad=0.2, bbox_to_anchor=(1.05, 0.02), 
        ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
    )
    
    SaveName = '{}_{:d}.{:d}_{}{}{}'.format(
        name, top_num, intv, 's' if smooth else '', 'f' if fill else '', 'd' if dot else '')
    plt.savefig(f'./AD/{SaveName}.svg', bbox_inches='tight')
    plt.show()


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

    nonZero = sum(df_use['SGLasso'].values > 0)
    df_use_sort = df_use.sort_values(['SGLasso'], ascending=False, kind='mergesort')
    df_use_sort.reset_index(drop=True, inplace=True)
    df_use_sort.insert(int(np.where(df_use_sort.columns == 'SGLasso')[0]) + 1, 'SGLasso_rank',
                       list(range(1, 1 + nonZero)) + [nonZero + 1] * (len(df_use) - nonZero))
    nonZero = sum(df_use['CDReg'].values > 0)
    df_use_sort = df_use_sort.sort_values(['CDReg'], ascending=False, kind='mergesort')
    df_use_sort.reset_index(drop=True, inplace=True)
    df_use_sort.insert(int(np.where(df_use_sort.columns == 'CDReg')[0]) + 1, 'CDReg_rank',
                       list(range(1, 1 + nonZero)) + [nonZero + 1] * (len(df_use) - nonZero))

    df_use_sort.to_csv('./AD/fea_slc.csv', index=False)

    df_use_sort = df_use_sort[(df_use_sort['CDReg_rank'] <= 15) | (df_use_sort['SGLasso_rank'] <= 15)]
    top_probes = df_use_sort[['fea_name', 'CHR', 'MAPINFO', 'gene_set',
                             'SGLasso', 'SGLasso_rank', 'CDReg', 'CDReg_rank']]
    top_probes.rename(columns={'fea_name': 'IlmnID'}, inplace=True)
    probe_pvals = pd.read_csv(os.path.join(data_root, 'pval_data', 'probe_pvals.csv'))
    top_probes = pd.merge(top_probes, probe_pvals, on='IlmnID')
    top_probes.rename(columns={'AD-NC': 'P-value'}, inplace=True)
    top_probes.to_csv('./AD/Top15sitesVS.csv', index=False)


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
    plt.savefig(f'./AD/{save_name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def DotPV(data, name=''):
    fig = plt.figure(figsize=(4/2.54, 21/2.54), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    a1 = sns.stripplot(
        ax=ax1, data=data, orient='h', x='logP', y='IlmnID',
        legend=False, edgecolor=None, color='grey',
    )
    ax1.set_xlim(0, 8)
    ax1.set_xticks([1, 4, 7])
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

    plt.savefig(f'./AD/{name}.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    main()

