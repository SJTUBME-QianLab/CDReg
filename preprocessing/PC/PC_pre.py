import os
import re
import random
import argparse
import time
import numpy as np
import pandas as pd
import gzip
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm import tqdm
from sklearn.impute import KNNImputer
import scipy
import scipy.stats
from statsmodels.stats.multitest import multipletests


def main():
    parser = argparse.ArgumentParser('Arguments Setting.')
    parser.add_argument('-sex', default=False, action='store_true')
    parser.add_argument('--depth', default=20, type=int)
    parser.add_argument('--na', default=0.3, type=float)
    parser.add_argument('--gp_min', default=20, type=int)
    parser.add_argument('--impute', default='Mean', type=str, choices=['KNN', 'EM', 'Mean'])
    parser.add_argument('--pv', default=0.05, type=float)
    parser.add_argument('--chr', type=int, default=20)
    args = parser.parse_args()
    bed_dir = "../raw_data/PC/bed_files"
    gencode_dir = "../raw_data/gencode_v27/gene_position/"
    chr = args.chr

    set_seed(2024)
    log_file = open('log.txt', 'a')
    print('\n\n', time.asctime(), args, file=log_file)
    save_dir = bed_dir + '_pre'
    info_csv = pd.read_excel("./Sample.xlsx", sheet_name="Sample")
    subj_dict = dict(zip(info_csv['Sample Name'], info_csv['Sample ID']))
    subj_dict_inv = {v: k for k, v in subj_dict.items()}

    # data matrix
    # matrix_met_d20_chr22_na0.3.csv
    matrix_name = f"matrix_met_d{args.depth}_{'chr22XY' if args.sex else 'chr22'}_na{args.na:g}.csv"
    print('Matrix Name:', matrix_name)
    if not os.path.isfile(os.path.join(save_dir, matrix_name)):
        matrix_met = merge_bed(bed_dir, subj_dict, args.depth,
                               'Y' if args.chr == ['Y'] else args.sex, args.na)
        matrix_met.to_csv(os.path.join(save_dir, matrix_name), index=True)
        print(time.asctime(), f'\tSave matrix ({matrix_met.shape}) done!', file=log_file)  # (10972421, 353)
        del matrix_met
    print('Matrix Exist:', os.path.join(save_dir, matrix_name), file=log_file)

    # Gene information
    gene_name = matrix_name.replace("matrix_met", "gene")
    if not os.path.isfile(os.path.join(save_dir, gene_name)):
        sites = add_gene_info(save_dir, matrix_name, gencode_dir)
        sites.to_csv(os.path.join(save_dir, gene_name), index=True)
        print(time.asctime(), f'\tAdd gene information {sites.shape} done!')  # (10972421, 6)
        del sites
    print('Gene Exist:', os.path.join(save_dir, gene_name), file=log_file)
    #    #chr  start    end       probe    gene  gene_num
    # 0  chr1  16555  16556  chr1:16555  WASH7P         1
    # 1  chr1  16570  16571  chr1:16570  WASH7P         1
    # 2  chr1  47176  47177  chr1:47176     NaN         0

    # single gene
    new_name = gene_name.replace('gene', 'sites')
    if not os.path.isfile(os.path.join(save_dir, new_name)):
        sites = pd.read_csv(os.path.join(save_dir, gene_name), index_col=0)
        print(time.asctime(), '\tOriginal sites:', len(sites), file=log_file)  # 10972421
        # single gene
        sites = sites[sites['gene_num'] == 1]
        print(time.asctime(), '\tSingle gene, length =', len(sites), file=log_file)  # 5674112
        # remove intersection
        sites, cross_gene = remove_intersect_MEM(sites)
        print(time.asctime(), '\tCross gene:\n', '; '.join(cross_gene), file=log_file)
        print(time.asctime(), '\tRemain:', sites.shape, ', length =', len(sites), file=log_file)  # 5673717
        print(time.asctime(), '\tChromosome length:\n',
              '; '.join([f"{i}: {len(sites[sites['#chr'] == f'chr{i}'])}"  for i in list(range(1, 23)) + ['X', 'Y']]),
              file=log_file)
        sites.to_csv(os.path.join(save_dir, new_name), index=True)
    print('Single Exist:', os.path.join(save_dir, new_name), file=log_file)

    # impute
    impute_name = matrix_name.replace('.csv',
                                      ('_EM.txt' if args.impute == 'EM' else f'_{args.impute}.csv'))
    if not os.path.isfile(os.path.join(save_dir, impute_name)):
        matrix_met = pd.read_csv(os.path.join(save_dir, matrix_name), index_col=0)
        if args.impute == 'Mean':
            Mean_impute(matrix_met, save_dir, impute_name)
        else:
            raise ValueError(f'Unknown impute method: {args.impute}.')
    print('Impute Exist:', os.path.join(save_dir, impute_name), file=log_file)

    # add label
    info_name = matrix_name.replace('.csv', '_info.csv')
    if not os.path.isfile(os.path.join(save_dir, info_name)):
        matrix_met = pd.read_csv(os.path.join(save_dir, matrix_name), index_col=0)
        label, df_pair, mask, info_csv = label_pair(info_csv, matrix_met.columns)
        # Finally, 162 * 2 = 324, + 28 = 352
        df_pair.to_csv(os.path.join(save_dir, matrix_name.replace('.csv', '_pairDF.csv')), index=True)
        np.save(os.path.join(save_dir, matrix_name.replace('.csv', '_pairMatrix.npy')), mask)
        info_csv.to_csv(os.path.join(save_dir, info_name), index=True)
    print('Label Exist:', os.path.join(save_dir, info_name), file=log_file)

    # p-value screen
    pval_name = impute_name.rsplit('.', 1)[0] + '_PVs.csv'
    print(time.asctime(), '\tread start')
    matrix_met1 = pd.read_csv(os.path.join(save_dir, impute_name), index_col=0)  # probes*samples
    sites0 = pd.read_csv(os.path.join(save_dir, new_name), index_col=0)
    print(time.asctime(), '\tread finish')

    chr_dir = os.path.join(save_dir, impute_name.rsplit('.', 1)[0])
    os.makedirs(chr_dir, exist_ok=True)
    pvi_name = pval_name.replace('.csv', f'CHR{chr}.csv')
    if os.path.isfile(os.path.join(chr_dir, pvi_name)):
        print(time.asctime(), '\tExist:', pvi_name, file=log_file)
    else:
        mat = matrix_met1.loc[sites0[sites0['#chr'] == f'chr{chr}']['probe'], :]
        print(time.asctime(), '\tCHR =', chr, 'probe number =', len(mat), file=log_file)
        pvals = calculate_p(save_dir, mat, info_name)
        pvals.to_csv(os.path.join(chr_dir, pvi_name), index=False)
        print(time.asctime(), '\tFinish calculate p-values', file=log_file)

    # final select
    sites_name = new_name.replace('.csv', f'_CHR{chr}p{args.pv}gp{args.gp_min}.csv')
    final_name = impute_name.rsplit('.', 1)[0] + f"CHR{chr}p{args.pv}gp{args.gp_min}.csv"

    pvals = pd.read_csv(os.path.join(chr_dir, pval_name.replace('.csv', f'CHR{chr}.csv')))
    sites_pv = pd.merge(sites0, pvals, on='probe', how='inner')  # 451214
    print(time.asctime(), '\tlen(pvals) =', len(pvals), ', len(sites) =', len(sites0), ', len(sites_pv) =', len(sites_pv), file=log_file)
    sites = sites_pv[sites_pv['pFDR'] < args.pv]  # 274630
    print(time.asctime(), '\tp.FDR <', args.pv, ', length =', len(sites), file=log_file)  # 274630
    # >= gp_min
    sites = extract_size_gt(sites, gp_min=args.gp_min)
    print(time.asctime(), '\tGroup size >=', args.gp_min, ', length =', len(sites), file=log_file)  # 266327
    tt = sites[['gene', 'gp_size', 'gp_idx0']].drop_duplicates()
    tt.index =  tt['gp_idx0']
    pd.testing.assert_series_equal(tt['gp_size'].astype(int), sites.groupby('gp_idx0', sort=False).size(),
                                   check_names=False)
    print(time.asctime(), '\tGene number =', len(tt), file=log_file)  # 1412
    del tt
    new_id = dict(zip(sites['gene'].unique(), range(len(sites['gene'].unique()))))
    sites['gp_idx'] = sites['gene'].map(new_id)
    sites.to_csv(os.path.join(chr_dir, sites_name), index=False)

    final_data = matrix_met1.loc[sites['probe'], :]
    final_data.to_csv(os.path.join(chr_dir, final_name), index=True)
    print(time.asctime(), f'\tFinally data.shape = {final_data.shape}', file=log_file)

    info = pd.read_csv(os.path.join(save_dir, info_name), index_col=0)

    X = final_data.values.T.astype(float)  # sample * feature
    nn = np.linalg.norm(X, ord=2, axis=0)
    X_norm = X / nn[None, :]  # n*d / 1*d
    np.save(os.path.join(chr_dir, final_name.replace('.csv', '_X_normL2.npy')), X_norm)
    np.savetxt(os.path.join(chr_dir, final_name.replace('.csv', '_X_normL2.csv')), X_norm, delimiter=',')
    info.index = info['Sample ID']
    Y = info.loc[final_data.columns, 'label'].values
    np.save(os.path.join(chr_dir, matrix_name.replace('.csv', '_Y.npy')), Y)
    np.savetxt(os.path.join(chr_dir, matrix_name.replace('.csv', '_Y.csv')), Y, delimiter=',')

    final_name = impute_name.rsplit('.', 1)[0] + f"CHR{chr}p{args.pv}gp{args.gp_min}.csv"
    print(time.asctime(), f'\tStart read bed {final_name}', file=log_file)
    final_data = pd.read_csv(os.path.join(chr_dir, final_name), index_col=0)
    samples = final_data.columns
    sites = final_data.index

    res = Parallel(n_jobs=8)(
        delayed(read_bed_gz_cut)(
        os.path.join(bed_dir, subj_dict_inv[subj] + '.bed.gz'), subj, sites)
        for subj in samples
    )
    met_all = pd.concat([kk[0] for kk in res], axis=1, join='outer')
    cov_all = pd.concat([kk[1] for kk in res], axis=1, join='outer')
    assert met_all.shape == cov_all.shape == (len(sites), len(samples))
    met_all = met_all[samples].loc[sites]
    cov_all = cov_all[samples].loc[sites]

    met_all.to_csv(os.path.join(chr_dir, final_name.replace('.csv', '_met.csv')), index=True)
    cov_all.to_csv(os.path.join(chr_dir, final_name.replace('.csv', '_cov.csv')), index=True)
    print(time.asctime(), f'\tFinish read bed {chr}', file=log_file)

    log_file.close()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def read_bed_gz(file_path, subj, depth, sex):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t')
    """
    0  #chr  start    end strand type total_num methy_num         percent_num
    1  chr1  10542  10543      +   CG         7         4  0.5714285714285714
    2  chr1  10563  10564      +   CG        10         8                 0.8
    3  chr1  10571  10572      +   CG        12        10  0.8333333333333334
    """
    # print(df.shape, os.path.split(file_path)[-1])
    df0 = df[(df['type'] == 'CG') & (df['total_num'].astype(int) >= depth)]
    if sex == 'Y':
        df0 = df0[df0['#chr'] == 'chrY']
    elif sex:
        df0 = df0[df0['#chr'].isin([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])]
    else:
        df0 = df0[df0['#chr'].isin([f'chr{i}' for i in range(1, 23)])]
    probes = [f"{k1}:{k2}" for k1, k2 in zip(df0['#chr'], df0['start'])]
    res = pd.Series(df0['percent_num'].values.astype(np.float16), index=probes)
    res.name = subj
    return res


def read_bed_gz_cut(file_path, subj, sites):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t')
    df0 = df[(df['type'] == 'CG')]
    df0.index = [f"{k1}:{k2}" for k1, k2 in zip(df0['#chr'], df0['start'])]
    met = pd.concat([pd.DataFrame(index=sites), df0['methy_num']], axis=1, join='inner')
    cov = pd.concat([pd.DataFrame(index=sites), df0['total_num']], axis=1, join='inner')
    met.columns = [subj]
    cov.columns = [subj]
    return met, cov


def merge_bed(bed_dir, subj_dict, depth, sex, na):
    matrix_met = pd.DataFrame()
    for i, (file, subj) in enumerate(subj_dict.items()):
        res = read_bed_gz(os.path.join(bed_dir, file + '.bed.gz'), subj, depth, sex)
        matrix_met = pd.concat([matrix_met, res], axis=1)
        print(f'{i+1}/{len(subj_dict)}', file, len(res))
    print(time.asctime(), '\tMerge data done!')

    # Calculate the proportion of NA in probes, and delete probes with NA content over 30%.
    na_proportion = matrix_met.isna().sum(axis=1) / matrix_met.shape[1]
    matrix_met.drop(na_proportion[na_proportion > na].index, axis=0, inplace=True)

    na_proportion = matrix_met.isna().sum(axis=0) / matrix_met.shape[0]
    matrix_met.drop(na_proportion[na_proportion > na].index, axis=1, inplace=True)
    print(time.asctime(), '\tDrop NA done!')

    # sort
    matrix_met['#chr'] = [kk.split(":")[0] for kk in matrix_met.index]
    matrix_met['start'] = [int(kk.split(":")[1]) for kk in matrix_met.index]
    matrix_met['chr'] = matrix_met['#chr'].apply(lambda x: int(x[3:]) if x[3:].isdigit() else x[3:])
    matrix_met.sort_values(by=['chr', 'start'], inplace=True)
    matrix_met.drop(['#chr', 'start', 'chr'], axis=1, inplace=True)

    return matrix_met


def Mean_impute(matrix_met, save_dir, new_name, cut=10):
    print(time.asctime(), '\tStart Mean impute')
    size = matrix_met.shape[0] // cut
    def sub(cc):
        sub_matrix = matrix_met.iloc[cc * size: min((cc + 1) * size, matrix_met.shape[0]), :]
        mean = sub_matrix.mean(axis=1)
        assert mean.values[0] == sub_matrix.iloc[0, :].mean()
        sub_matrix = sub_matrix.T.fillna(mean).T
        return sub_matrix
    res = Parallel(n_jobs=3)(delayed(sub)(cc) for cc in tqdm(range(cut + 1)))
    matrix_met = pd.concat(res, axis=0)
    matrix_met.to_csv(os.path.join(save_dir, new_name), index=True)
    print(time.asctime(), '\tFinish Mean impute')


def label_pair(info_csv, data_columns):
    info_csv = pd.concat([info_csv[info_csv['Sample ID'] == kk] for kk in data_columns], axis=0).reset_index(drop=True)
    label_dict = {'Adjacent': 0, 'Prostate Cancer': 1}
    label = info_csv['Healthy Condition'].map(label_dict)
    pd.testing.assert_series_equal(label,
                                   info_csv['Tissue/Cell Line'].map({'Prostate gland': 0, 'Prostate gland cancer': 1}),
                                   check_names=False)
    label.index = info_csv['Sample ID']

    patient = []
    for i in range(len(info_csv)):
        description = info_csv.loc[i, 'Description']
        if '(T2N0M0) from 70-year-old male named T502' in description:
            pat = 'T503'
        elif '(T1cN0M0) from 67-year-old male named T67' in description:
            pat = 'T68'
        else:
            pat = description.split('named ')[1].split(' ')[0]
        patient.append(pat)

    assert [1 if kk[0] == 'T' else 0 for kk in patient] == label.values.tolist()
    info_csv.insert(info_csv.columns.get_loc('Description') + 1, 'patient', [int(kk[1:]) for kk in patient])
    info_csv.insert(info_csv.columns.get_loc('patient') + 1, 'label', label.values)
    sample_patient = info_csv[['Sample ID', 'patient', 'label']]
    df_dup = info_csv[info_csv['patient'].duplicated(keep=False)]

    patient_list = sorted(df_dup['patient'].unique())
    print(len(df_dup), len(patient_list))  # 324 162
    outlier = []
    df_pair = []
    for pp in patient_list:
        ll = df_dup[df_dup['patient'] == pp]
        ll.sort_values(by='label', inplace=True)
        if set(ll['label']) == {0, 1}:
            df_pair.append(ll)
        else:
            outlier.append(ll)

    # df_pair：duplicate and paired
    # outlier：duplicate and not paired
    if len(outlier):
        outlier = pd.concat(outlier, axis=0)
        print(outlier)
    else:
        print('There is no outlier.')
    # There is no outlier.

    df_pair = pd.concat(df_pair, axis=0)
    print(df_pair.shape)  # (366, 3)
    assert sum(df_pair['label'] == 0) == sum(df_pair['label'] == 1)

    # construct mask
    count_pair = 0
    count_sample = 0.0
    mask = np.zeros((len(info_csv), len(info_csv)))
    for i in range(0, len(df_pair), 2):
        k1, k2 = df_pair.index[i:(i + 2)]
        assert set(df_pair.loc[[k1, k2], 'label']) == {0, 1}
        assert df_pair.loc[k1, 'Sample ID'] == data_columns[k1]
        count_pair += 1
        count_sample += 2
        mask[k1, k2] = 1.0
        mask[k2, k1] = 1.0

    print('Finally, %d patients, %d samples' % (len(df_pair)/2, count_sample))
    # Finally, 162 patients, 324 samples
    assert count_pair * 2 == np.sum(mask)

    return label, df_pair, mask, info_csv


def calculate_p(save_dir, matrix_met, info_name):
    print(time.asctime(), '\tStart calculate p-values')
    info_csv = pd.read_csv(os.path.join(save_dir, info_name), index_col=0)
    label = info_csv[['Sample ID', 'label']]
    label.index = label['Sample ID']
    matrix_met1 = pd.concat([label[['label']], matrix_met.T], axis=1)  # samples*(1+probes)
    pos = matrix_met1[matrix_met1['label'] == 1].iloc[:, 1:].T
    neg = matrix_met1[matrix_met1['label'] == 0].iloc[:, 1:].T

    def sub(pr):
        pos_i = pos.loc[pr, :]
        neg_i = neg.loc[pr, :]
        pv = TTestPV(pos_i, neg_i)
        return pr, pv
    res = Parallel(n_jobs=-1, timeout=99999)(delayed(sub)(pr) for pr in tqdm(matrix_met.index))
    # res = Parallel(n_jobs=10)(delayed(sub)(pr) for pr in tqdm(list(matrix_met.index)[:10]))

    pvals = pd.DataFrame({
        'probe': [kk[0] for kk in res],
        'pvalue': [kk[1] for kk in res],
    })
    reject, corrected_p_values, _, _ = multipletests(pvals['pvalue'], method='fdr_bh')
    pvals['pFDR'] = corrected_p_values
    print(time.asctime(), '\tFinish calculate p-values')
    return pvals


def TTestPV(aa, bb, alternative="two-sided"):
    var = scipy.stats.levene(aa, bb).pvalue
    pv = scipy.stats.ttest_ind(aa, bb, equal_var=(var > 0.05)).pvalue
    if alternative == "two-sided":
        return pv
    elif alternative in ["greater", "less"]:
        return pv / 2
    else:
        raise ValueError(f"Unknown alternative: {alternative}.")


def get_sites(bed_dir, matrix_name):
    print(time.asctime(), '\tStart read matrix')
    matrix_met = pd.read_csv(os.path.join(bed_dir, matrix_name), index_col=0)
    print(time.asctime(), '\tFinish read matrix')
    sites = pd.DataFrame([kk.split(":") for kk in matrix_met.index], columns=['#chr', 'start'])
    sites['start'] = sites['start'].astype(int)
    sites['end'] = sites['start'].astype(int) + 1
    sites['probe'] = [f"{k1}:{k2}" for k1, k2 in zip(sites['#chr'], sites['start'])]
    return sites


def add_gene_info(bed_dir, matrix_name, gencode_dir):
    print(time.asctime(), '\tStart read matrix')
    matrix_met = pd.read_csv(os.path.join(bed_dir, matrix_name), index_col=0)
    print(time.asctime(), '\tFinish read matrix')
    sites = pd.DataFrame([kk.split(":") for kk in matrix_met.index], columns=['#chr', 'start'])
    sites['start'] = sites['start'].astype(int)
    sites['end'] = sites['start'].astype(int) + 1
    sites['probe'] = [f"{k1}:{k2}" for k1, k2 in zip(sites['#chr'], sites['start'])]

    # GENCODE - Human Release 27 (GRCh38.p10)
    # Use `gencode_v27.sh` to get `*.position` files
    # vers = re.search(r'gencode_v(\d+)', gencode_dir).group(1)
    gencode = pd.read_csv(os.path.join(gencode_dir, "allGene.hg38.position"), sep='\t', header=None)
    #       0      1      2            3
    # 0  chr1  11869  14409      DDX11L1
    # 1  chr1  14404  29570       WASH7P
    # 2  chr1  17369  17436    MIR6859-1
    gencode.columns = ['#chr', 'gene_start', 'gene_end', 'gene']
    gencode = gencode[gencode['#chr'].isin([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])]
    gencode['gene_start'] = gencode['gene_start'].astype(int)
    gencode['gene_end'] = gencode['gene_end'].astype(int)

    sites_gene = pd.DataFrame({
        'probe': sites['probe'].values,
        'gene': [''] * len(sites),
    })
    def catch_sites(g):
        chr, start, end, gene = gencode.loc[g, :]
        slc = (sites['#chr'] == chr) & (sites['start'] >= start) & (sites['start'] < end)
        for i in sites_gene.loc[slc].index:
            ex = sites_gene.loc[i, 'gene']
            if ex == '':
                sites_gene.loc[i, 'gene'] = gene
            elif gene not in ex.split(';'):
                sites_gene.loc[i, 'gene'] = ex + ';' + gene
            else:
                pass
    # Parallel(n_jobs=10, require='sharedmem')(delayed(catch_sites)(g) for g in tqdm(range(10)))
    Parallel(n_jobs=10, require='sharedmem')(delayed(catch_sites)(g) for g in tqdm(range(len(gencode))))
    sites = pd.merge(sites, sites_gene, on='probe', how='left')
    print(sites.iloc[:20, :], sites.iloc[-20:, :])
    print(time.asctime(), '\tMerge gene & sites done!')

    assert (sites['probe'] == matrix_met.index).all()
    a1 = [len(x.split(';')) for x in sites['gene'] if isinstance(x, str)]
    a2 = [len(set(x.split(';'))) for x in sites['gene'] if isinstance(x, str)]
    assert a1 == a2

    sites['gene_num'] = sites['gene'].apply(lambda x: len(x.split(';')) if (isinstance(x, str) and x != '') else 0)

    return sites


def extract_size_gt(pr_gene, gp_min=5):
    # group larger than gp_min
    uu = pr_gene.groupby(['gene'], sort=False).sum()
    uu = uu[uu['gene_num'] >= gp_min]

    # concat information
    group = pd.DataFrame({
        'gene': uu.index,
        'gp_size': uu['gene_num'].values,
        'gp_idx0': range(len(uu)),
    })
    info = pd.merge(pr_gene, group, on='gene', how='left')  # keep sort
    info.dropna(inplace=True)
    return info


def remove_intersect_MEM(info):
    # 创建一个字典，跟踪每个基因的最后一次索引
    last_seen = {}
    cross_gene = []
    info.reset_index(drop=True, inplace=True)

    for i in tqdm(range(1, len(info))):
    # for i in tqdm(range(1, 10)):
        current_gene = info['gene'][i]
        if current_gene in cross_gene:
            continue
        elif current_gene in last_seen.keys():
            if i != last_seen[current_gene][-1] + 1:  # 出现过的基因, 但和之前不连续
                cross_gene.append(current_gene)
                print(current_gene, i, last_seen[current_gene])
            tmp = last_seen[current_gene]
            tmp.append(i)
            last_seen[current_gene] = tmp
        else:
            last_seen[current_gene] = [i]  # 更新最后一次索引

    sites = info[~info['gene'].isin(cross_gene)]
    gp_idx = dict(zip(sites['gene'].unique(), range(len(sites['gene'].unique()))))
    sites['gp_idx'] = sites['gene'].map(gp_idx)
    s1 = sites.sort_values(by=['gp_idx'], kind='stable')
    assert list(s1.index) == list(sites.index)
    sites.drop(['gp_idx'], axis=1, inplace=True)
    return sites, cross_gene


if __name__ == '__main__':
    main()