import re

import pandas as pd
import numpy as np
import os
import seaborn as sns
import sklearn.metrics
import scipy
import pickle

from joblib import Parallel, delayed
from itertools import product, permutations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.dpi'] = 900
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

eval_root = './simulation/'
os.makedirs(eval_root, exist_ok=True)
data_root = './../data/simulation/'
result_root = './../results/'
# data_root = '/home/data/tangxl/ContrastSGL/sim_data2/'
# result_root = '/home/data/tangxl/ContrastSGL/results2'

method_dict = {
    'LASSO': 'Lasso', 'Enet0.8': 'ENet', 'SGLASSO': 'SGLasso', 'pclogit': 'Pclogit',
    'L10.4L210.05Ls0Lc0.1_lr0.0001': 'CDReg w/o S',
    'L10.4L210.05Ls1.2Lc0_lr0.0001': 'CDReg w/o C',
    'L10.4L210.05Ls1.2Lc0.1_lr0.0001': 'CDReg',
}
colorsPal = sns.color_palette('Pastel1')
colors_dict = dict(zip(
    list(method_dict.values()),
    colorsPal[1:5] + colorsPal[6:8] + [colorsPal[0]]
))

metric_dict = {
    'AUPRC': 'AUPRC', 'AUROC': 'AUROC', 'MCC': 'MCC',
}
data_dict = {
    'covGau_de1_b0.5': 'Gau',
    'covLap2_de1_b0.5': 'Lap2',
}


def main():
    if not os.path.isfile(os.path.join(eval_root, 'all_results.csv')):
        RawMetric()

    if not os.path.isfile(os.path.join(eval_root, 'statistics.csv')):
        df_all = pd.read_csv(os.path.join(eval_root, 'all_results.csv'))
        df_all.rename(columns={'roc_auc': 'auc', 'indi_abs_w_mean': 'indi'}, inplace=True)
        MetricSet = dict()
        for data_name in data_dict.keys():
            dfi = df_all[df_all['data'] == data_name]
            for metric in ['AUROC', 'AUPRC', 'MCC', 'indi', 'isol']:
                vv = CalMetric(dfi, metric)
                MetricSet[f"{data_dict[data_name]}_{metric}"] = vv
        with open(os.path.join(eval_root, 'MetricSet.pkl'), 'wb') as f:
            pickle.dump(MetricSet, f)

        compilations = concat_metrics(MetricSet)
        compilations.to_csv(os.path.join(eval_root, 'statistics.csv'), index=True)

    # Existing files: all_results.csv, MetricSet.pkl, statistics.csv

    data = pd.read_csv(os.path.join(eval_root, 'all_results.csv'))
    with open(os.path.join(eval_root, 'MetricSet.pkl'), 'rb') as f:
        MetricSet = pickle.load(f)
    head = 'Fig3b'
    config = {
        'xlim': [0.45, 0.75], 'xticks': [0.5, 0.6, 0.7], 'x0': 0.75, 'head': head,
    }
    plot_MetricBar(data, MetricSet, 'AUROC', config, pv='Ttest')
    config = {
        'xlim': [0, 0.325], 'xticks': [0.1, 0.2, 0.3], 'x0': 0.3, 'head': head,
    }
    plot_MetricBar(data, MetricSet, 'AUPRC', config, pv='Ttest')
    config = {
        'xlim': [0, 0.27], 'xticks': [0, 0.1, 0.2], 'x0': 0.26, 'head': head,
    }
    plot_MetricBar(data, MetricSet, 'MCC', config, pv='Ttest')

    plot_box_indi(data.copy(), 'Fig3c')
    plot_box_isol(data.copy(), 'Fig3c')

    # heatmaps of weights
    data_name = 'covGau_de1_b0.5'
    df1 = GetWeightIndi(data_name, 2027)
    df1s = GetWeightIsol(data_name, 2027)
    data_name = 'covLap2_de1_b0.5'
    df2 = GetWeightIndi(data_name, 2027)
    df2s = GetWeightIsol(data_name, 2027)
    pd_writer = pd.ExcelWriter(os.path.join(eval_root, 'Fig3d.xlsx'))
    df1.to_excel(pd_writer, sheet_name='d-indi_set1', index=False)
    df1s.to_excel(pd_writer, sheet_name='d-isol_set1', index=False)
    df2.to_excel(pd_writer, sheet_name='d-indi_set2', index=False)
    df2s.to_excel(pd_writer, sheet_name='d-isol_set2', index=False)
    pd_writer.save()
    HeatmapIndiOne(df1, head='Fig3d', add=True)
    HeatmapIndiOne(df2, head='Fig3d', add=False)
    HeatmapIsolOne(df1s, head='Fig3d', add=True)
    HeatmapIsolOne(df2s, head='Fig3d', add=False)

    if not os.path.isfile(os.path.join(eval_root, 'variations.csv')):
        concat_variation()
    df_lap2 = pd.read_csv(os.path.join(eval_root, 'variations.csv'))
    model_name = pd.read_csv(os.path.join(eval_root, 'variations_name.csv'), index_col=0)
    RVave, RVstd, xy_data = RelativeVariation(df_lap2, model_name, 'acc')
    ParamBubble(xy_data.copy(), head='FigS4', metric='acc')
    xy_data.to_csv(os.path.join(eval_root, 'FigS4.csv'), index=False)

    # simulated correlation
    head = 'covGau_de1_b0.5_seed2027'
    dist = Simulate_corr(head)
    ScatterDistCorr(dist, head='FigS5', setting=head[3:6])
    head = 'covLap2_de1_b0.5_seed2027'
    dist = Simulate_corr(head)
    ScatterDistCorr(dist, head='FigS5', setting=head[3:6])

    # Table remake
    reset = pd.DataFrame()
    data = pd.read_csv(os.path.join(eval_root, 'all_results.csv'))
    for data_name in data_dict.keys():
        dfi = data[data['data'] == data_name]
        for metric in ['AUROC', 'AUPRC', 'MCC', 'indi', 'isol']:
            cut = pd.DataFrame({
                mm: dfi[dfi['method'] == mm][metric].values for mm in method_dict.values()
            })
            cut.columns = method_dict.values()
            print(cut.shape)
            cut['data'] = data_name
            cut['metric'] = metric
            reset = pd.concat([reset, cut], axis=0)
    reset.to_csv(os.path.join(eval_root, 'Table_remake.csv'), index=False)


def RawMetric():
    metrics_seeds = []
    for pre in data_dict.keys():
        data_seeds = [int(kk.split('_seed')[1]) for kk in sorted(os.listdir(result_root)) if kk.startswith(pre)]
        for ss in data_seeds:
            res = concat_my(pre, ss)
            metrics_seeds.append(res)
    metrics_seeds = pd.concat(metrics_seeds, axis=0).reset_index(drop=True)
    metrics_seeds['method'] = metrics_seeds['method'].apply(lambda x: method_dict[x])
    metrics_seeds.to_csv(os.path.join(eval_root, 'all_results.csv'), index=False)


def concat_my(pre, seed, method_list=None):
    if method_list is None:
        method_list = method_dict.keys()

    head = ['data_seed', 'method', 's']
    data_name = f'{pre}_seed{seed}'
    df_rand = []
    for mm in method_list:
        try:
            a = [int(kk.split('s')[1]) for kk in os.listdir(os.path.join(result_root, data_name, mm))]
        except:
            print(os.path.join(result_root, data_name, mm))
            exit()
        for s in [int(kk.split('s')[1]) for kk in os.listdir(os.path.join(result_root, data_name, mm))]:
            root_dir = os.path.join(result_root, data_name, mm, f's{s}')
            re = GetResults(root_dir)
            if not re.eval:
                continue
            df_rand.append(pd.concat([pd.DataFrame([seed, mm, s], index=head).T, re.onetime(choice=1)], axis=1))
    df_rand = pd.concat(df_rand, axis=0)
    df_rand['data'] = pre
    return df_rand


class GetResults:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_name = 'eval_FS1.xlsx'
        if os.path.isfile(os.path.join(self.root_dir, self.file_name)):
            self.eval = True
        else:
            self.eval = False

    def acc_allT(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='acc_allT')
        return eval_FS

    def metrices(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='metrices')
        final_weight = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='final_weight')
        prob_1 = final_weight['abs_weight_normalize']
        true = final_weight['TorF'].values
        pre, rec, _ = sklearn.metrics.precision_recall_curve(true, prob_1)
        eval_FS.loc[0, 'AUPRC'] = sklearn.metrics.auc(rec, pre)
        eval_FS.loc[0, 'AUROC'] = sklearn.metrics.roc_auc_score(true, prob_1)
        return eval_FS

    def group_check(self):
        evals = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='group_check1')
        # inef = evals.iloc[0:30, -1].mean(axis=0)
        true = evals.iloc[30:48, :]['abs_weight_normalize'].mean(axis=0)
        isol = evals.iloc[48:66, :]['abs_weight_normalize'].mean(axis=0)

        evals = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='group_check2')
        ineff = evals.iloc[[k + i * 22 for i in range(3) for k in range(19, 22)], :]['abs_weight_normalize'].mean(axis=0)  # 8,9,10
        bias = evals.iloc[[k + i * 22 for i in range(3) for k in [7]], :]['abs_weight_normalize'].mean(axis=0)  # 7
        group_mean = pd.DataFrame([true, isol, ineff, bias], index=['true', 'isol', 'inef', 'bias'])
        return group_mean.T

    def group_check_rank(self):
        evals = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='group_check1')
        # inef = evals.iloc[0:30, -1].mean(axis=0)
        true = evals.iloc[30:48, :][['rank_ave', 'rank_min']].mean(axis=0)
        isol = evals.iloc[48:66, :][['rank_ave', 'rank_min']].mean(axis=0)

        evals = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='group_check2')
        ineff = evals.iloc[[k + i * 22 for i in range(3) for k in range(19, 22)], :][['rank_ave', 'rank_min']].mean(axis=0)  # 8,9,10
        bias = evals.iloc[[k + i * 22 for i in range(3) for k in [7]], :][['rank_ave', 'rank_min']].mean(axis=0)  # 7
        # group_mean = pd.DataFrame([true, isol, ineff, bias], index=['true', 'isol', 'inef', 'bias'])
        group_mean = pd.concat([true, isol, ineff, bias], axis=0)
        group_mean.index = [k1+k2 for k1, k2 in product(['true', 'isol', 'inef', 'bias'], ['Ave', 'Min'])]
        return group_mean

    def true_std(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='df_true_std')
        evals = eval_FS.loc[:, 'abs_weight_normalize'].mean(axis=0)
        return pd.DataFrame({'true_std': [evals]})

    def true_std_rank(self):
        eval_FS = pd.read_excel(os.path.join(self.root_dir, self.file_name), sheet_name='df_true_std')
        evals = eval_FS.loc[:, ['rank_ave', 'rank_min']].mean(axis=0)
        evals.index = ['true_stdAve', 'true_stdMin']
        return evals

    def onetime(self, choice=0):
        if choice == 0:
            return pd.concat([self.metrices(), self.group_check(), self.true_std()], axis=1)
        elif choice == 1:
            res = pd.concat([self.metrices(), self.group_check(), self.true_std()], axis=1)
            add = pd.concat([self.group_check_rank(), self.true_std_rank()], axis=0)
            res = pd.concat([res, pd.DataFrame(add).T], axis=1)
            # print(res)
        else:
            raise ValueError(f'wrong choice : {choice}')
        return res


def concat_metrics(MetricSet):
    compilations = []
    for kk, vv in MetricSet.items():
        dd = vv.copy()
        dt, met = kk.split('_')
        dd['data'] = dt
        dd['metric'] = met
        compilations.append(dd)
    compilations = pd.concat(compilations, axis=0)
    return compilations


def Rep10Values(df, method, metric):
    dfi = df[df['method'] == method]
    dfi = dfi[metric]
    return dfi.values


def TTestPV(aa, bb, alternative='two-sided'):
    # assert alternative in ['two-sided', 'greater', 'less']
    # return scipy.stats.ttest_rel(aa, bb, alternative=alternative).pvalue
    if alternative in ['greater', 'less']:
        return scipy.stats.ttest_rel(aa, bb).pvalue / 2
    elif alternative == 'two-sided':
        return scipy.stats.ttest_rel(aa, bb).pvalue
    else:
        raise ValueError(alternative)


def TTestPV_ind(aa, bb):
    var = scipy.stats.levene(aa, bb).pvalue
    return scipy.stats.ttest_ind(aa, bb, equal_var=(var > 0.05)).pvalue


def CalMetric(df, metric):
    if metric in ['acc', 'AUROC', 'AUPRC']:
        side = 'greater'
    elif metric in ['indi', 'isol']:
        side = 'less'
    else:
        raise ValueError(metric)
    order = method_dict.values()
    dfV = df[['method', metric]]
    cal = pd.concat([
        dfV.groupby(['method']).mean().T.rename(index={metric: 'mean'}),
        dfV.groupby(['method']).std(ddof=0).T.rename(index={metric: 'std'}),
    ], axis=0)
    cal.loc['Ttest'] = 0.
    ours = Rep10Values(df, 'CDReg', metric)
    for mm in cal.columns:
        if mm == 'CDReg':
            continue
        bb = Rep10Values(df, mm, metric)
        cal.loc['Ttest', mm] = TTestPV(ours, bb, side)

    return cal.loc[:, order]


def plot_MetricBar(data, MetricSet, met, config, pv=False):
    print(met, data[met].min(), data[met].max(), config['xlim'])
    # colors0 = sns.color_palette('Pastel1')
    # colors = colors0[1:5] + colors0[6:8] + [colors0[0]]

    if met in ['acc']:
        fig = plt.figure(figsize=((config['xlim'][1] - config['xlim'][0])*100, 5))
    else:
        fig = plt.figure(figsize=(2.5, 5))
    ax1 = plt.subplot(1, 1, 1)

    bar1 = sns.barplot(
        ax=ax1, data=data, y="data", x=met, orient='h',
        errorbar='se',
        order=['covGau_de1_b0.5', 'covLap2_de1_b0.5'], 
        hue='method', hue_order=method_dict.values(),
        palette=colors_dict, width=0.85,
    )
    str1 = sns.stripplot(
        ax=ax1, data=data, y="data", x=met, orient='h',
        order=['covGau_de1_b0.5', 'covLap2_de1_b0.5'],
        hue='method', hue_order=method_dict.values(), dodge=True,
        palette=['grey']*len(method_dict), size=4, edgecolor="0.2", linewidth=0.5, alpha=0.5
    )

    ax1.set_xlim(*config['xlim'])
    ax1.set_xticks(config['xticks'])
    ax1.set_xlabel('')

    ax1.set_ylim(1.5, -0.5)
    ax1.set_ylabel('')
    ax1.set_yticks(ticks=[0, 1], labels=['Setting 1', 'Setting 2'],
                   rotation=90, verticalalignment='center')
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
        bar1.bar_label(i, fmt='%.3f', padding=-40, fontsize=10)

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
        for i, mm in enumerate(list(method_dict.values())[:-1]):
            ax1.text(x0, start + width * (i+0.5), 
                     '{:.2e}'.format(MetricSet[f'Gau_{met}'].loc[pv, mm]),
                    horizontalalignment='center', verticalalignment='center')
            ax1.text(x0, start + width * (i+0.5) + 1, 
                     '{:.2e}'.format(MetricSet[f'Lap2_{met}'].loc[pv, mm]),
                    horizontalalignment='center', verticalalignment='center')

    if met == 'acc':
        plot_method_legend(colors_dict, os.path.join(eval_root, f"{config['head']}_legend_rows.svg"))
        plot_method_legend(colors_dict, os.path.join(eval_root, f"{config['head']}_legend_cols.svg"), cols=-1)
        # ax1.legend(
        #     loc='center right', fontsize=10, bbox_to_anchor=(-0.15, 0.5), # borderaxespad=0.2,
        #     ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
        # )
    ax1.legend_.remove()

    fig.savefig(os.path.join(eval_root,f"{config['head']}_{met}.svg"), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def plot_method_legend(ColorSet, out_path, cols=1):
    size=12
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=cc) for mm, cc in ColorSet.items()]
    leg_config = dict(
        borderaxespad=0.2,
        labelspacing=0.4,
        handlelength=0.8,
        columnspacing=0.8,
        handletextpad=0.4,
        borderpad=0,
        frameon=False
    )
    fig_m = plt.figure(figsize=(3, 3))
    fig_m.set_facecolor('None')
    plt.figlegend(
        handles, ColorSet.keys(),
        ncol=len(ColorSet) if cols == -1 else cols,
        fontsize=size-2,
        loc='center', # bbox_to_anchor=(1.01, 0.5),
        **leg_config
    )
    fig_m.tight_layout()
    fig_m.savefig(out_path, bbox_inches='tight', pad_inches=0.02)


def plot_box_indi(data, head):
    def sub_ax(dfi, ax1):
        box1 = sns.boxplot(
            ax=ax1, data=dfi, y="method", x="indi", order=method_order, orient='h', 
            # whis=[0, 100],
            palette=colors_dict,
        )
        str1 = sns.stripplot(
            ax=ax1, data=dfi, y="method", x="indi", order=method_order, orient='h', 
            size=4, color="0.8", edgecolor="0.2", linewidth=0.5, alpha=0.5
        )
        ax1.tick_params(
            axis='both', labelsize=10, length=2, width=1.5,
            bottom=True, top=False, left=True, right=False,
            labelbottom=True, labeltop=False, labelleft=True, labelright=False,
        )

        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['top'].set_visible(False)

        ax1.set_facecolor('None')
        ax1.set_xlabel('')
        ax1.set_xticks(ticks=np.linspace(start=0, stop=0.1, num=6, endpoint=True))
        sns.despine(
            ax=ax1, top=True, right=True, left=False, bottom=False, 
            offset=3, trim=True
        )
        ax1.yaxis.set_label_coords(-0.56, 0)
        ax1.yaxis.set_label_position('right')
    
    # colors0 = sns.color_palette('Pastel1')
    # colors = colors0[1:5] + [colors0[7]] + [colors0[0]]
    method_order = [kk for kk in method_dict.values() if kk != 'CDReg w/o S']

    fig = plt.figure(figsize=(2.5, 5))
    ax1 = plt.subplot(2, 1, 1)
    df1 = data[data['data']=='covGau_de1_b0.5']
    sub_ax(df1, ax1)
    ax1.set_ylabel('Setting 1', fontsize=12)
    
    ax2 = plt.subplot(2, 1, 2)
    df2 = data[data['data']=='covLap2_de1_b0.5']
    sub_ax(df2, ax2)
    ax2.set_ylabel('Setting 2', fontsize=12)
    ax1.set_title('Weight of subject-specific sites', fontsize=12, weight='bold')
    
    plt.savefig(os.path.join(eval_root, f'{head}_indi.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def plot_box_isol(data, head):
    def sub_ax(dfi, ax1):
        box1 = sns.boxplot(
            ax=ax1, data=dfi, y="method", x="isol", order=method_order, orient='h', 
            # whis=[0, 100],
            palette=colors_dict,
        )
        str1 = sns.stripplot(
            ax=ax1, data=dfi, y="method", x="isol", order=method_order, orient='h', 
            size=4, color="0.8", edgecolor="0.2", linewidth=0.5, alpha=0.5
        )
        ax1.tick_params(
            axis='both', labelsize=10, length=2, width=1.5,
            bottom=True, top=False, left=True, right=False,
            labelbottom=True, labeltop=False, labelleft=True, labelright=False,
        )

        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['top'].set_visible(False)

        ax1.set_facecolor('None')
        ax1.set_xlabel('')
        ax1.set_xticks(ticks=np.linspace(start=0, stop=0.1, num=6, endpoint=True))
        sns.despine(
            ax=ax1, top=True, right=True, left=False, bottom=False, 
            offset=3, trim=True
        )
        ax1.yaxis.set_label_coords(-0.56, 0)
        ax1.yaxis.set_label_position('right')

    # colors0 = sns.color_palette('Pastel1')
    # colors = colors0[1:5] + [colors0[6]] + [colors0[0]]
    method_order = [kk for kk in method_dict.values() if kk != 'CDReg w/o C']

    fig = plt.figure(figsize=(2.5, 5))
    ax1 = plt.subplot(2, 1, 1)
    df1 = data[data['data']=='covGau_de1_b0.5']
    sub_ax(df1, ax1)
    
    ax2 = plt.subplot(2, 1, 2)
    df2 = data[data['data']=='covLap2_de1_b0.5']
    sub_ax(df2, ax2)
    
    ax1.set_ylabel('Setting 1', fontsize=12)
    ax2.set_ylabel('Setting 2', fontsize=12)

    ax1.set_title('Weight of isolated sites', fontsize=12, weight='bold')
    
    plt.savefig(os.path.join(eval_root, f'{head}_isol.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def GetWeightIndi(data_name, seed):
    method_order = [kk for kk in method_dict.keys() if method_dict[kk] != 'CDReg w/o S']
    indi_idx = np.load(os.path.join(data_root, f'{data_name}_seed{seed}', 'spac_idx.npy'))
    result_path0 = os.path.join(result_root, f'{data_name}_seed{seed}')
    fs_indi_dict = {}
    for mm in method_order:
        s = [int(kk.split('s')[1]) for kk in os.listdir(os.path.join(result_path0, mm))]
        assert len(s) == 1
        s = s[0]
        fs = pd.read_excel(os.path.join(result_path0, mm, f's{int(s)}', 'eval_FS.xlsx'),
                           sheet_name='final_weight')
        fs_indi = fs.iloc[indi_idx, :]['abs_weight_normalize']
        fs_indi_dict[method_dict[mm]] = fs_indi
    head = fs.iloc[indi_idx, :5]
    out = pd.concat([head, pd.DataFrame(fs_indi_dict)], axis=1)
    return out


def GetWeightIsol(data_name, seed):
    method_order = [kk for kk in method_dict.keys() if method_dict[kk] != 'CDReg w/o C']
    basic = pd.read_csv(os.path.join(data_root, f'{data_name}_seed{seed}', 'basic_info.csv'))
    isol_idx = basic[basic['isol_01'] == 1].index
    result_path0 = os.path.join(result_root, f'{data_name}_seed{seed}')
    fs_indi_dict = {}
    for mm in method_order:
        s = [int(kk.split('s')[1]) for kk in os.listdir(os.path.join(result_path0, mm))]
        assert len(s) == 1
        s = s[0]
        fs = pd.read_excel(os.path.join(result_path0, mm, f's{int(s)}', 'eval_FS.xlsx'),
                           sheet_name='final_weight')
        fs_indi = fs.iloc[isol_idx, :]['abs_weight_normalize']
        fs_indi_dict[method_dict[mm]] = fs_indi
    head = fs.iloc[isol_idx, :5]
    out = pd.concat([head, pd.DataFrame(fs_indi_dict)], axis=1)
    return out


def HeatmapIndiOne(dfi, head, add=False):
    fig = plt.figure(figsize=(3.2, 2.5))
    gs = fig.add_gridspec(
        2, 2,  width_ratios=(20, 1), height_ratios=(12, 1),
        wspace=0.05, hspace=0.05
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_chr = fig.add_subplot(gs[1, 0], sharex=ax1)

    color_dict = {7: '#ECE4AB', 17: '#C6E0B0', 27: '#EAE9E8'}

    xx = dfi.iloc[:, 5:].values.T
    print(xx.min(), xx.max())
    a1 = sns.heatmap(dfi.iloc[:, 5:].T, ax=ax1, linecolor='white', linewidth=1, 
                     vmin=0, vmax=xx.max(), cmap=plt.cm.Blues, cbar_ax=ax_cbar)

    for spine in ax1.spines.values():
        spine.set(visible=True, lw=1, edgecolor="black")

    ax1.tick_params(
        axis='both', labelsize=10, 
        length=2, width=1.5,
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )

    # gene patches
    color_dict0 = dict()
    for num in color_dict.keys():
        scope = np.where(dfi['ch']==num)[0]
        print(num, scope)
        color_dict0[num] = [color_dict[num], scope.min(), scope.max()]

    sns.heatmap(dfi[['ch']].T, ax=ax_chr, annot=False,
                cmap=[vv[0] for vv in color_dict0.values()], cbar=False)
    ax_chr.set_xticks([0] + [vv[2] + 1 for vv in color_dict0.values()])
    ax_chr.tick_params(
        axis='both',
        length=20, width=1, direction='in',
        bottom=False, top=True, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )
    ax_chr.set_yticklabels(['Gene'], rotation=0, fontsize=10)
    for kk, vv in color_dict0.items():
        loc = (vv[1] + vv[2])/2
        # ax_chr.text(
        #     loc, 0.5, str(kk), rotation=90,
        #     horizontalalignment='left', verticalalignment='center',
        #     fontsize=5
        # )
        print(loc)
        ax_chr.text(
            loc+0.5, 1.1, str(kk),
            horizontalalignment='center', verticalalignment='top',
            fontsize=10
        )
    
    print(ax1.get_ylim())
    ax1.yaxis.set_label_coords(-0.46, 3)
    ax1.yaxis.set_label_position('right')
    
    if add:
        ax1.set_ylabel('Setting 1', fontsize=12)
        ax1.set_title('Weight of subject-specific sites', fontsize=12, weight='bold')
        plt.savefig(os.path.join(eval_root, f'{head}_indi_set1.svg'), bbox_inches='tight', pad_inches=0.01)
    else:
        ax1.set_ylabel('Setting 2', fontsize=12)
        plt.savefig(os.path.join(eval_root, f'{head}_indi_set2.svg'), bbox_inches='tight', pad_inches=0.01)

    # plt.show()


def group_consecutive(numbers):
    result = []
    current_group = []

    for number in numbers:
        if not current_group or number == current_group[-1] + 1:
            current_group.append(number)
        else:
            result.append(current_group)
            current_group = [number]

    if current_group:
        result.append(current_group)

    return result


def HeatmapIsolOne(dfi, head, add=False):
    fig = plt.figure(figsize=(3.2, 2.5))
    gs = fig.add_gridspec(
        2, 2,  width_ratios=(20, 1), height_ratios=(12, 1),
        wspace=0.05, hspace=0.05
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_chr = fig.add_subplot(gs[1, 0], sharex=ax1)

    colors0 = sns.color_palette('Set3')  # 12个
    colors1 = sns.color_palette('Set2')  # 8个
    colors = colors0[2:] + colors1
    colors = colors[:6] + ['white'] * 4 + colors[6:12] + ['white'] * 4 + colors[12:]

    xx = dfi.iloc[:, 5:].values.T
    print(xx.min(), xx.max())
    a1 = sns.heatmap(dfi.iloc[:, 5:].T, ax=ax1, linecolor='white', linewidth=1, 
                     vmin=0, vmax=xx.max(), cmap=plt.cm.Blues, cbar_ax=ax_cbar)

    for spine in ax1.spines.values():
        spine.set(visible=True, lw=1, edgecolor="black")

    ax1.tick_params(
        axis='both', labelsize=10, 
        length=2, width=1.5,
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )

    # gene patches
    color_dict0 = dict()
    for i, num in enumerate(sorted(set(dfi['ch']))):
        scope = np.where(dfi['ch']==num)[0]
        color_dict0[num] = [colors[i], scope.min(), scope.max()]

    sns.heatmap(dfi[['ch']].T, ax=ax_chr, annot=False, 
                cmap=colors, alpha=0.8, cbar=False)
    groups = group_consecutive(color_dict0.keys())
    ax_chr.set_xticks([0] + [color_dict0[gg[-1]][2] + 1 for gg in groups])
    ax_chr.tick_params(
        axis='both',
        length=20, width=1, direction='in',
        bottom=False, top=True, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )
    ax_chr.set_yticklabels(['Gene'], rotation=0, fontsize=10)
    # for kk, vv in color_dict0.items():
    #     loc = (vv[1] + vv[2])/2
    #     ax_chr.text(
    #         loc, 0.5, str(kk), rotation=90,
    #         horizontalalignment='left', verticalalignment='center',
    #         fontsize=5
    #     )

    for gg in groups:
        g_min = color_dict0[gg[0]][1]
        g_max = color_dict0[gg[-1]][2]
        loc = (g_min + g_max)/2
        print(loc)
        ax_chr.text(
            loc+0.5, 1.1, f"{gg[0]}-{gg[-1]}",
            horizontalalignment='center', verticalalignment='top',
            fontsize=10
        )

    print(ax1.get_ylim())
    ax1.yaxis.set_label_coords(-0.46, 3)
    ax1.yaxis.set_label_position('right')
    
    if add:
        ax1.set_ylabel('Setting 1', fontsize=12)
        ax1.set_title('Weight of isolated sites', fontsize=12, weight='bold')
        plt.savefig(os.path.join(eval_root, f'{head}_isol_set1.svg'), bbox_inches='tight', pad_inches=0.01)
    else:
        ax1.set_ylabel('Setting 2', fontsize=12)
        plt.savefig(os.path.join(eval_root, f'{head}_isol_set2.svg'), bbox_inches='tight', pad_inches=0.01)

    # plt.show()


def concat_variation():
    raw_name = list(method_dict.keys())[-1]
    match = re.search('L1(\d+\.\d+)L21(\d+\.\d+)Ls(\d+\.\d+)Lc(\d+\.\d+)_lr0.0001', raw_name)
    L1, L21, Ls, Lc = [float(kk) for kk in match.groups()]
    model_name = pd.DataFrame(
        columns=['L1', 'L21', 'Ls', 'Lc'],
        index=[-0.1, -0.05, 0.05, 0.1],
    )
    for j in model_name.columns:
        for i in model_name.index:
            var_str = '{:g}'.format((1 + i) * float(eval(j)))
            model_name.loc[i, j] = raw_name.replace(f'{j}{eval(j)}', f'{j}{var_str}')

    pre = 'covLap2_de1_b0.5'
    metrics_seeds = []
    data_seeds = [int(kk.split('_seed')[1]) for kk in sorted(os.listdir(result_root)) if kk.startswith(pre)]
    method_list = [raw_name] + model_name.values.flatten().tolist()
    for ss in data_seeds:
        res = concat_my(pre, ss, method_list=method_list)
        metrics_seeds.append(res[['data_seed', 'method', 'acc']])
    metrics_seeds = pd.concat(metrics_seeds, axis=0).reset_index(drop=True)
    metrics_seeds.to_csv(os.path.join(eval_root, 'variations.csv'), index=False)
    model_name.to_csv(os.path.join(eval_root, 'variations_name.csv'), index=True)


def RelativeVariation(df, model_name, metric):
    target = df[df['method']==list(method_dict.keys())[-1]][metric].values
    RVave = pd.DataFrame(index=model_name.index, columns=model_name.columns)
    RVstd = pd.DataFrame(index=model_name.index, columns=model_name.columns)
    for j in model_name.columns:
        for i in model_name.index:
            dfi = df[df['method']==model_name.loc[i, j]][metric].values
            RV = (dfi - target) / target * 100
            RVave.loc[i, j] = RV.mean()
            RVstd.loc[i, j] = RV.std()
    xy_data = []
    for var in RVave.index:
        for mm in RVave.columns:
            xy_data.append([var, mm, RVave.loc[var, mm], RVstd.loc[var, mm]])
    xy_data = pd.DataFrame(xy_data, columns=['var', 'method', f'RV{metric}', f'RV{metric}_std'])
    return RVave, RVstd, xy_data


def ParamBubble(data, head, metric):
    colormap = plt.cm.coolwarm
    
    fig = plt.figure(figsize=(3.2, 2.4))
    
    vmin, vmax = data[f'RV{metric}'].min(), data[f'RV{metric}'].max()
    normalize = mcolors.CenteredNorm(vcenter=0)
    try:
        data['var'] = data['var'].apply(lambda x: '{:g}%'.format(float(x) * 100))
    except ValueError:
        pass
    
    ax1 = plt.subplot(1, 1, 1)
    a1 = ax1.scatter(
        x=data['var'], y=data['method'], c=data[f'RV{metric}'].values, s=data[f'RV{metric}_std'].values*1000,
        cmap=colormap, norm=normalize,
        edgecolor='black', linewidths=1, 
    )
    cb = fig.colorbar(
        a1, ax=ax1, 
        ticks=np.array([-0.15, -0.1, -0.05, 0, 0.05]), 
        boundaries=np.linspace(vmin, vmax, 300, endpoint=True),
    )
    cb.ax.tick_params(labelsize=10)
    
    ax1.tick_params(
        axis='both', labelsize=12, length=0, #width=1.5,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    sns.despine(
        ax=ax1, top=True, right=True, left=True, bottom=True, 
        offset=5, trim=True,
    )
    ax1.grid(True, linewidth=1, alpha=0.8)
    ax1.set_axisbelow('True')

    ax1.set_facecolor('None')
    ax1.set_ylim(3.4, -0.4)
    ax1.set_xlim(-0.4, 3.4)
    ax1.set_xlabel('Fluctuation', fontsize=12)
    ax1.set_ylabel('Hyperparameter', fontsize=12)
    ax1.set_yticks([0, 1, 2, 3], ['$\lambda_1$', '$\lambda_2$', '$\lambda_S$', '$\lambda_C$'])

    plt.savefig(os.path.join(eval_root, f'{head}_stability.png'), bbox_inches='tight', pad_inches=0.01)
    plt.savefig(os.path.join(eval_root, f'{head}_stability.svg'), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def Simulate_corr(head):
    X = np.load(os.path.join(data_root, head, 'X.npy'))
    basic_info = pd.read_csv(os.path.join(data_root, head, 'basic_info.csv'))
    df = []
    for ch in range(1, 1+30):
        group = basic_info[basic_info['gp_label']==ch]
        data = X[:, group.index]
        num = data.shape[1]
        for i in range(num - 1):
            dist = group.iloc[i+1, :]['loc'] - group.iloc[i, :]['loc']
            corr = np.corrcoef(data[:, i], data[:, i+1])[0, 1]
            df.append([dist, corr])
    df = pd.DataFrame(df, columns=['dist', 'corr'])
    return df


def ScatterDistCorr(data, head, setting='', frac=1):
    dot_size = 20

    fig = plt.figure(figsize=(3.2, 2.4), facecolor='none')
    ax1 = plt.subplot(1, 1, 1)

    data0 = data.copy()
    if frac < 1:
        data0 = data0.sample(frac=frac, random_state=2023).reset_index(drop=True)
        print(len(data0), len(data))

    a1 = sns.scatterplot(
        ax=ax1, data=data0, x='dist', y='corr',
        marker='.', s=dot_size, color='#8CA29A',
    )

    min_corr = data0['corr'].min() - 0.05
    ax1.set_ylim(min(min_corr, 0), 1.05)
    if min_corr < 0:
        ax1.axhline(y=0, xmin=0, xmax=205, linewidth=1, color='black', linestyle='--')
    ax1.set_xlim(0, 205)
    ax1.set_xticks([0, 50, 100, 150, 200])
    ax1.tick_params(
        axis='both', labelsize=12, length=2,
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    )
    ax1.set_xlabel('Distance between adjacent sites', fontsize=12)
    ax1.set_ylabel('Pearson correlation', fontsize=12)
    if setting == 'Gau':
        ax1.set_title('Setting 1 (Gaussian)', fontsize=12, weight='bold')
    elif setting == 'Lap':
        ax1.set_title('Setting 2 (Laplace)', fontsize=12, weight='bold')

    ax1.set_facecolor('None')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    # ax1.spines['left'].set_position('zero')
    # ax1.spines['bottom'].set_position('zero')
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)

    plt.savefig(os.path.join(eval_root, f'{head}_ScatterDistCorr_f{frac}_{setting}.svg'),
                bbox_inches='tight', pad_inches=0.01)
    data0.to_csv(os.path.join(eval_root, f'{head}_ScatterDistCorr_f{frac}_{setting}.csv'), index=False)
    # plt.show()


if __name__ == '__main__':
    main()



