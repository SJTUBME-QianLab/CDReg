import pandas as pd
import numpy as np
import os
import seaborn as sns
import sklearn.metrics
import scipy
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

plt.rcdefaults()
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 900  # 图片像素
plt.rcParams['figure.dpi'] = 900  # 分辨率
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


method_dict = {
    'LASSO': 'Lasso',
    'Enet0.8': 'ENet',
    'SGLASSO': 'SGLasso',
    'pclogit': 'Pclogit',
    'L10.4L210.05Ls0Lc0.1_lr0.0001': 'CDReg w/o S',
    'L10.4L210.05Ls1.2Lc0_lr0.0001': 'CDReg w/o C',
    'L10.4L210.05Ls1.2Lc0.1_lr0.0001': 'CDReg',
}
data_dict = {
    'covGau_de1_b0.5': 'Gau',
    'covLap2_de1_b0.5': 'Lap2',
}
os.makedirs('./simulation', exist_ok=True)
eval_root = './'
data_root = './../data/simulation/'
result_root = './../results/'


def main():
    # acc and auc
    df_gau1 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='Gau1-rep10')
    df_gau1 = df_gau1.iloc[:, :12].dropna(axis=0, inplace=False)
    df_gau1['method'] = df_gau1['method'].apply(lambda x: method_dict[x])
    df_Lap2 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='Lap2-rep10')
    df_Lap2 = df_Lap2.iloc[:, :12].dropna(axis=0, inplace=False)
    df_Lap2['method'] = df_Lap2['method'].apply(lambda x: method_dict[x])
    df_all = pd.concat([df_gau1, df_Lap2], axis=0)
    df_all.rename(columns={'roc_auc': 'auc', 'indi_abs_w_mean': 'indi'}, inplace=True)

    MetricSet = dict()
    for data_name in data_dict.keys():
        dfi = df_all[df_all['data'] == data_name]
        for metric in ['acc', 'auc', 'indi', 'isol']:
            vv = CalMetric(dfi, metric)
            MetricSet[f"{data_dict[data_name]}_{metric}"] = vv

    compilations = concat_metrics(MetricSet)
    compilations.to_csv('./simulation/statistics.csv', index=True)

    data = df_all[['data_seed', 'method', 's', 'acc', 'auc', 'indi', 'isol', 'data']]

    plot_AUC(data, MetricSet, pv='Ttest')
    plot_Acc(data, MetricSet, pv='Ttest')
    plot_box_indi(data.copy())
    plot_box_isol(data.copy())

    # heatmaps of weights
    data_name = 'covGau_de1_b0.5'
    seeds = df_all[df_all['data'] == data_name][['data_seed', 'method', 's']]
    df1 = GetWeightIndi(data_name, 2027, seeds)
    df1s = GetWeightIsol(data_name, 2027, seeds)
    data_name = 'covLap2_de1_b0.5'
    seeds = df_all[df_all['data'] == data_name][['data_seed', 'method', 's']]
    df2 = GetWeightIndi(data_name, 2027, seeds)
    df2s = GetWeightIsol(data_name, 2027, seeds)

    HeatmapIndiOne(df1, add=True)
    HeatmapIndiOne(df2, add=False)
    HeatmapIsolOne(df1s, add=True)
    HeatmapIsolOne(df2s, add=False)

    # parameter sensibility
    df_lap2 = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='stability-Lap2')
    model_name = pd.read_excel(os.path.join(eval_root, 'Results_Simu.xlsx'), sheet_name='stability-name')
    model_name.index = model_name['variation']
    model_name.drop(['variation'], axis=1, inplace=True)
    RVacc, RVacc_s = RelativeVariation(df_lap2, 'acc', model_name)

    xy_data = []
    for var in RVacc.index:
        for mm in RVacc.columns:
            xy_data.append([var, mm, RVacc.loc[var, mm], RVacc_s.loc[var, mm]])
    xy_data = pd.DataFrame(xy_data, columns=['var', 'method', 'RVAcc', 'RVAcc_std'])
    ParamBubble(xy_data.copy())


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
    # return stats.ttest_rel(aa, bb, alternative=alternative).pvalue
    if alternative in ['greater', 'less']:
        return stats.ttest_rel(aa, bb).pvalue / 2
    else:
        return stats.ttest_rel(aa, bb).pvalue


def TTestPV_ind(aa, bb):
    var = scipy.stats.levene(aa, bb).pvalue
    return scipy.stats.ttest_ind(aa, bb, equal_var=(var > 0.05)).pvalue  # p值大于0.05，说明满足方差相等


def CalMetric(df, metric):
    if metric in ['acc', 'auc']:
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

    return cal.loc[:, order].T


def plot_AUC(data, MetricSet, pv=None):
    colors0 = sns.color_palette('Pastel1')
    colors = colors0[1:5] + colors0[6:8] + [colors0[0]]

    fig = plt.figure(figsize=(3, 5))
    ax1 = plt.subplot(1, 1, 1)

    bar1 = sns.barplot(
        ax=ax1, data=data, y="data", x="auc", orient='h', width=0.85,
        order=['covGau_de1_b0.5', 'covLap2_de1_b0.5'],
        hue='method', hue_order=method_dict.values(), palette=colors,
    )

    ax1.set_xlim(0.93, 0.965)
    ax1.set_xticks([0.93, 0.94, 0.95, 0.96])
    ax1.set_xlabel('')
    ax1.set_ylim(1.5, -0.5)
    ax1.set_yticks(ticks=[0, 1], labels=[])
    ax1.set_ylabel('')

    ax1.set_title('AUC', size=12, weight='bold')
    
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
        bar1.bar_label(i, fmt='%.3f', padding=-50, fontsize=10)

    if pv is not None:
        x0 = 0.966
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
        for i in range(6):
            ax1.text(x0, start + width * (i+0.5), 
                     '{:.2e}'.format(MetricSet['Gau_auc'][pv].values[i]),
                    horizontalalignment='center', verticalalignment='center')
        for i in range(6):
            ax1.text(x0, start + width * (i+0.5) + 1, 
                     '{:.2e}'.format(MetricSet['Lap2_auc'][pv].values[i]),
                    horizontalalignment='center', verticalalignment='center')

    ax1.legend_.remove()
    
    plt.savefig('./simulation/Fig3b_AUC.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_Acc(data, MetricSet, pv=False):
    colors0 = sns.color_palette('Pastel1')
    colors = colors0[1:5] + colors0[6:8] + [colors0[0]]

    fig = plt.figure(figsize=(3/3.5*2.5, 5))
    ax1 = plt.subplot(1, 1, 1)

    bar1 = sns.barplot(
        ax=ax1, data=data, y="data", x="acc", orient='h', width=0.85,
        order=['covGau_de1_b0.5', 'covLap2_de1_b0.5'], 
        hue='method', hue_order=method_dict.values(), palette=colors,
    )

    ax1.set_xlim(0.92, 0.945)
    ax1.set_xticks([0.92, 0.93, 0.94])
    ax1.set_xlabel('')

    ax1.set_ylim(1.5, -0.5)
    ax1.set_yticks(ticks=[0, 1], labels=['Setting 1', 'Setting 2'], 
                   rotation=90, verticalalignment='center')
    ax1.set_ylabel('')

    ax1.set_title('Accuracy', size=12, weight='bold')

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
        bar1.bar_label(i, fmt='%.3f', padding=-50, fontsize=10)

    if pv is not None:
        x0 = 0.946
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
        for i in range(6):
            ax1.text(x0, start + width * (i+0.5), 
                     '{:.2e}'.format(MetricSet['Gau_acc'][pv].values[i]),
                    horizontalalignment='center', verticalalignment='center')
        for i in range(6):
            ax1.text(x0, start + width * (i+0.5) + 1, 
                     '{:.2e}'.format(MetricSet['Lap2_acc'][pv].values[i]),
                    horizontalalignment='center', verticalalignment='center')
    
    ax1.legend(
        loc='center right', fontsize=10, bbox_to_anchor=(-0.15, 0.5), # borderaxespad=0.2, 
        ncol=1, handlelength=1, columnspacing=0.5, handletextpad=0.5, borderpad=0.5
    )
    
    plt.savefig('./simulation/Fig3b_ACC.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_box_indi(data):
    def sub_ax(dfi, ax1):
        box1 = sns.boxplot(
            ax=ax1, data=dfi, y="method", x="indi", order=method_order, orient='h', 
            whis=[0, 100], palette=colors, 
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
    
    colors0 = sns.color_palette('Pastel1')
    colors = colors0[1:5] + [colors0[7]] + [colors0[0]]
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
    
    plt.savefig('./simulation/Fig3c_indi.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_box_isol(data):
    def sub_ax(dfi, ax1):
        box1 = sns.boxplot(
            ax=ax1, data=dfi, y="method", x="isol", order=method_order, orient='h', 
            whis=[0, 100], palette=colors,
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

    colors0 = sns.color_palette('Pastel1')
    colors = colors0[1:5] + [colors0[6]] + [colors0[0]]
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
    
    plt.savefig('./simulation/Fig3c_isol.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


def GetWeightIndi(data_name, seed, df_seeds):
    method_order = [kk for kk in method_dict.keys() if method_dict[kk] != 'CDReg w/o S']
    indi_idx = np.load(os.path.join(data_root, f'{data_name}_seed{seed}', 'spac_idx.npy'))
    result_path0 = os.path.join(result_root, f'{data_name}_seed{seed}')
    fs_indi_dict = {}
    for mm in method_order:
        s = df_seeds[(df_seeds['data_seed'] == seed) & (df_seeds['method'] == method_dict[mm])]['s'].values
        fs = pd.read_excel(os.path.join(result_path0, mm, f's{int(s)}', 'eval_FS.xlsx'),
                           sheet_name='final_weight')
        fs_indi = fs.iloc[indi_idx, :]['abs_weight_normalize']
        fs_indi_dict[method_dict[mm]] = fs_indi
    head = fs.iloc[indi_idx, :5]
    out = pd.concat([head, pd.DataFrame(fs_indi_dict)], axis=1)
    return out


def GetWeightIsol(data_name, seed, df_seeds):
    method_order = [kk for kk in method_dict.keys() if method_dict[kk] != 'CDReg w/o C']
    basic = pd.read_csv(os.path.join(data_root, f'{data_name}_seed{seed}', 'basic_info.csv'))
    isol_idx = basic[basic['isol_01'] == 1].index
    result_path0 = os.path.join(result_root, f'{data_name}_seed{seed}')
    fs_indi_dict = {}
    for mm in method_order:
        s = df_seeds[(df_seeds['data_seed'] == seed) & (df_seeds['method'] == method_dict[mm])]['s'].values
        fs = pd.read_excel(os.path.join(result_path0, mm, f's{int(s)}', 'eval_FS.xlsx'),
                           sheet_name='final_weight')
        fs_indi = fs.iloc[isol_idx, :]['abs_weight_normalize']
        fs_indi_dict[method_dict[mm]] = fs_indi
    head = fs.iloc[isol_idx, :5]
    out = pd.concat([head, pd.DataFrame(fs_indi_dict)], axis=1)
    return out


def HeatmapIndiOne(dfi, add=False):
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

    # chromosome patches
    color_dict0 = dict()
    for num in color_dict.keys():
        scope = np.where(dfi['ch']==num)[0]
        color_dict0[num] = [color_dict[num], scope.min(), scope.max()]

    sns.heatmap(dfi[['ch']].T, ax=ax_chr, annot=False, 
                cmap=[vv[0] for vv in color_dict0.values()], cbar=False)
    ax_chr.tick_params(
        axis='both', labelsize=10, 
        length=2, width=1.5,
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )
    ax_chr.set_yticklabels(['Chromosome'], rotation=0, fontsize=10)
    for kk, vv in color_dict0.items():
        loc = (vv[1] + vv[2])/2
        ax_chr.text(
            loc, 0.5, str(kk), rotation=90,
            horizontalalignment='left', verticalalignment='center', 
            fontsize=5
        )
    
    print(ax1.get_ylim())
    ax1.yaxis.set_label_coords(-0.46, 3)
    ax1.yaxis.set_label_position('right')
    
    if add:
        ax1.set_ylabel('Setting 1', fontsize=12)
        ax1.set_title('Weight of subject-specific sites', fontsize=12, weight='bold')
        plt.savefig('./simulation/Fig3d_indi_set1.svg', bbox_inches='tight', pad_inches=0.01)
    else:
        ax1.set_ylabel('Setting 2', fontsize=12)
        plt.savefig('./simulation/Fig3d_indi_set2.svg', bbox_inches='tight', pad_inches=0.01)

    plt.show()


def HeatmapIsolOne(dfi, add=False):
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

    # chromosome patches
    color_dict0 = dict()
    for i, num in enumerate(sorted(set(dfi['ch']))):
        scope = np.where(dfi['ch']==num)[0]
        color_dict0[num] = [colors[i], scope.min(), scope.max()]
    
    sns.heatmap(dfi[['ch']].T, ax=ax_chr, annot=False, 
                cmap=colors, alpha=0.8, cbar=False)
    ax_chr.tick_params(
        axis='both', labelsize=10, 
        length=2, width=1.5,
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=True, labelright=False,
    )
    ax_chr.set_yticklabels(['Chromosome'], rotation=0, fontsize=10)
    for kk, vv in color_dict0.items():
        loc = (vv[1] + vv[2])/2
        ax_chr.text(
            loc, 0.5, str(kk), rotation=90,
            horizontalalignment='left', verticalalignment='center', 
            fontsize=5
        )
    
    print(ax1.get_ylim())
    ax1.yaxis.set_label_coords(-0.46, 3)
    ax1.yaxis.set_label_position('right')
    
    if add:
        ax1.set_ylabel('Setting 1', fontsize=12)
        ax1.set_title('Weight of isolated sites', fontsize=12, weight='bold')
        plt.savefig('./simulation/Fig3d_isol_set1.svg', bbox_inches='tight', pad_inches=0.01)
    else:
        ax1.set_ylabel('Setting 2', fontsize=12)
        plt.savefig('./simulation/Fig3d_isol_set2.svg', bbox_inches='tight', pad_inches=0.01)

    plt.show()


def RelativeVariation(df, metric, model_name):
    target = df.loc[range(10), metric].values
    RVave = pd.DataFrame(index=model_name.index, columns=model_name.columns)
    RVstd = pd.DataFrame(index=model_name.index, columns=model_name.columns)
    for xx in ['L1', 'L21', 'S', 'C']:
        for yy in ['-10%', '-5%', '+5%', '+10%']:
            dfi = df[df['method']==model_name.loc[yy, xx]][metric].values
            RV = (dfi - target) / target * 100
            RVave.loc[yy, xx] = RV.mean()
            RVstd.loc[yy, xx] = RV.std()
    return RVave, RVstd


def ParamBubble(data):
    data['method'] = [kk.replace('STV', 'SRR') for kk in data['method']]
    colormap = plt.cm.coolwarm
    
    fig = plt.figure(figsize=(3.2, 2.4))
    
    vmin, vmax = data['RVAcc'].min(), data['RVAcc'].max()
    normalize = mcolors.CenteredNorm(vcenter=0)
    
    ax1 = plt.subplot(1, 1, 1)
    a1 = ax1.scatter(
        x=data['var'], y=data['method'], c=data['RVAcc'].values, s=data['RVAcc_std'].values*1000, 
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
    
    plt.savefig('./simulation/Figs1_stability.png', bbox_inches='tight', pad_inches=0.01)
    plt.savefig('./simulation/Figs1_stability.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    main()



