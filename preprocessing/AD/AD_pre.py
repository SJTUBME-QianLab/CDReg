import pandas as pd
import numpy as np
import os
import time
import re


def main():
    out_dir = './../../data/AD/'
    # raw_dir = './../../raw_data/AD/'
    out_dir0 = '/'
    raw_dir = '/'
    loc_dir = os.path.join(out_dir, 'HumanMethylation850_probe_chr_loc_sort.csv')
    chr_loc(os.path.join(raw_dir, 'MethylationEPIC_v-1-0_B4.csv'), loc_dir)  # get loc.csv file

    gp_min = 5
    # p_name = 'nc_ad_p0.05_96349.csv'
    find = [x for x in os.listdir(os.path.join(out_dir0, 'pval_data'))
            if re.match(r'nc_ad_p0.05_\d+.csv', x)]
    assert len(find) == 1
    p_name = find[0]
    pval = p_name.split('_')[2][1:]
    # ############ group by gene ############
    data = pd.read_csv(os.path.join(out_dir0, 'pval_data', p_name))
    os.makedirs(os.path.join(out_dir, 'group_gene'), exist_ok=True)
    info1, final_data = extract_1gene_probes_850k(loc_dir, os.path.join(out_dir, 'group_gene'), data, pval, gp_min)
    info1.to_csv(os.path.join(out_dir, 'group_gene', 'info_p%s.csv' % pval), index=False)
    final_data.to_csv(os.path.join(out_dir, 'group_gene', 'data_p%s_1gene_gt%d.csv' % (pval, gp_min)), index=False)

    # ############ barcode pair ############
    anno = pd.read_excel(os.path.join(raw_dir, 'ADNI_DNA_Methylation_SampleAnnotation_20170530.xlsx'),
                         sheet_name='SampleAnnotation')
    final_data = pd.read_csv(os.path.join(out_dir, 'group_gene', 'data_p%s_1gene_gt%d.csv' % (pval, gp_min)))
    df_pair, mask = get_pair(final_data.iloc[:1, 1:], anno)
    # Finally, 9 patients, 27 samples, 46 pairs
    df_pair.to_csv(os.path.join(out_dir, 'group_gene', 'pair_barcode.csv'), index=True)
    np.save(os.path.join(out_dir, 'group_gene', 'pair_matrix.npy'), mask)

    # ############ normalize ############
    final_data = pd.read_csv(os.path.join(out_dir, 'group_gene', 'data_p%s_1gene_gt%d.csv' % (pval, gp_min)))
    X = final_data.iloc[1:, 1:].values.T.astype(float)  # sample * feature
    nn = np.linalg.norm(X, ord=2, axis=0)
    X_norm = X / nn[None, :]  # n*d / 1*d
    Y = (final_data.iloc[0, 1:].values/2).astype(int)  # NC=0,AD=2 ==> NC=0,AD=1
    print(X_norm.shape, Y.shape)
    np.save(os.path.join(out_dir, 'group_gene', 'X_normL2_p%sgt%d.npy' % (pval, gp_min)), X_norm)
    np.save(os.path.join(out_dir, 'group_gene', 'Y.npy'), Y)
    np.savetxt(os.path.join(out_dir, 'group_gene', 'X_normL2.csv'), X_norm, delimiter=',')
    np.savetxt(os.path.join(out_dir, 'group_gene', 'Y.csv'), Y, delimiter=',')


def extract_1gene_probes_850k(loc_dir, out_dir, data1, pval, gp_min=5):
    assert data1.columns[0] == 'probe'
    assert data1.iloc[0, 0] == 'label_NC0_MCI1_AD2'
    data1.rename(columns={'probe': 'IlmnID'}, inplace=True)
    data1.iloc[0, 0] = '-1'
    data1.iloc[0, 1:] = data1.iloc[0, 1:].astype(int)
    data1.iloc[1:, 1:] = data1.iloc[1:, 1:].astype(float)

    if os.path.isfile(os.path.join(out_dir, 'allprobe_gene_850k_p%s.csv' % pval)):
        print('exist allprobe_gene.csv')
        pr_gene = pd.read_csv(os.path.join(out_dir, 'allprobe_gene_850k_p%s.csv' % pval))
    else:
        data1_use = data1.iloc[:, :1]

        # add chromosome (CHR) and location (MAPINFO)
        loc = pd.read_csv(loc_dir)
        loc = loc[['IlmnID', 'UCSC_RefGene_Name', 'CHR', 'MAPINFO']]
        loc['CHR'] = [(int(float(x)) if ((x not in ['X', 'Y', np.nan]) and (x == x)) else x) for x in loc['CHR']]
        loc.loc[-1] = ['-1', '-1', -1, -1]
        loc.index = loc.index + 1
        loc = loc.sort_index()

        pr_gene = pd.merge(loc, data1_use, on='IlmnID', how='inner')
        pr_gene.insert(1, 'gene_set', np.nan)
        pr_gene.insert(2, 'gene_num', np.nan)

        for i in range(1, len(pr_gene)):
            if pd.isna(pr_gene.loc[i, 'UCSC_RefGene_Name']):
                pr_gene.loc[i, 'gene_num'] = 0
                continue
            else:
                print(i)
                gene_set = sorted(list(set(pr_gene.loc[i, 'UCSC_RefGene_Name'].split(';'))))
                pr_gene.loc[i, 'gene_set'] = ';'.join(gene_set)
                pr_gene.loc[i, 'gene_num'] = len(gene_set)

        pr_gene.iloc[0, 1] = '-1'
        pr_gene.iloc[0, 2] = 1
        os.makedirs(out_dir, exist_ok=True)
        pr_gene.to_csv(os.path.join(out_dir, 'allprobe_gene_850k_p%s.csv' % pval), index=False)
        print('save allprobe_gene_850k_p%s.csv' % pval)

    # corresponding to only one gene
    pr_gene_use = pr_gene[pr_gene['gene_num'] == 1]
    pr_gene_use.reset_index(drop=True, inplace=True)

    # group larger than gp_min
    uu = pr_gene_use.groupby(['gene_set']).sum()
    uu = uu[uu['gene_num'] >= gp_min]

    # concat information
    tt = list(pr_gene_use['gene_set'].drop_duplicates())
    group = pd.DataFrame({
        'gene_set': tt,
        'gp_size': [-1] + [(uu.loc[x, 'gene_num'] if x in uu.index else np.nan) for x in tt[1:]],
        'gp_idx': [-1] + list(range(len(tt) - 1))
    })
    info = pd.merge(pr_gene_use, group, on='gene_set', how='left')  # keep sort
    info.dropna(inplace=True)
    info.reset_index(drop=True, inplace=True)

    # remove intersect genes, and correct gp_idx
    kk = -1
    gene_i = info.loc[0, 'gene_set']
    for i in range(1, len(info)):
        if info.loc[i, 'gene_set'] != gene_i:  # different gene
            if info.loc[i, 'gene_set'] in set(info.loc[:(i - 1), 'gene_set']):  # if the gene is duplicated
                print('intersect! %d' % i)
                info.loc[i, 'gp_idx'] = np.nan
                continue
            else:
                kk += 1  # group number +1
                gene_i = info.loc[i, 'gene_set']
                info.loc[i, 'gp_idx'] = kk
        else:  # same gene
            info.loc[i, 'gp_idx'] = kk

    info1 = info.dropna(inplace=False).reset_index(drop=True, inplace=False)

    if info1.groupby(['gene_set']).sum()[1:]['gene_num'].min() < gp_min:
        # further -- group larger than gp_min
        uu = info1.groupby(['gene_set']).sum()
        drop_gene = list(uu.index[uu['gene_num'] < gp_min][1:])  # ['MIR548H3']

        tt = [item for item in list(info1['gene_set'].drop_duplicates()) if item not in drop_gene]
        group = pd.DataFrame({'gene_set': tt})

        info2 = pd.merge(info1, group.dropna(), on='gene_set', how='inner')  # keep sort
        info2.reset_index(drop=True, inplace=True)

        kk = -1
        gene_i = info2.loc[0, 'gene_set']
        for i in range(1, len(info2)):
            if info2.loc[i, 'gene_set'] != gene_i:  # different gene
                kk += 1  # 组号+1
                gene_i = info2.loc[i, 'gene_set']  # update gene
                info2.loc[i, 'gp_idx'] = kk
            else:  # 与当前基因相同
                info2.loc[i, 'gp_idx'] = kk

        assert max(info2['gp_idx']) + 2 == len(group.dropna(inplace=False))
        info1 = info2.copy()
        del info2

    for i in range(2, len(info1)):
        if info1.loc[i, 'CHR'] < info1.loc[i - 1, 'CHR']:
            print('wrong: %d' % i)
        if info1.loc[i, 'CHR'] == info1.loc[i - 1, 'CHR'] and info1.loc[i, 'MAPINFO'] < info1.loc[i - 1, 'MAPINFO']:
            print('wrong: %d' % i)

    final_data = pd.merge(info1[['IlmnID']], data1, on='IlmnID')
    print('group num: %d, site num: %d' % (max(info1['gp_idx']) + 1, final_data.shape[0]-1))

    return info1, final_data


def get_pair(data, anno):
    sample_name = pd.DataFrame({
        'barcodes': [x[1:] for x in data.columns],
        'label': data.iloc[0, :]
    })
    sample_patient = pd.merge(sample_name, anno, on='barcodes')

    df_dup = sample_patient[sample_patient['RID'].duplicated(keep=False)]

    patient_list = sorted(list(set(df_dup['RID'])))
    outlier = []
    df_pair = []
    for pp in patient_list:
        ll = df_dup[df_dup['RID'] == pp]
        if set(ll['label']) == {0.0, 2.0}:
            df_pair.append(ll)
            print(len(ll))
        else:
            outlier.append(ll)

    # df_pair：duplicate and paired
    # outlier：duplicate and not paired
    if len(outlier):
        outlier = pd.concat(outlier, axis=0)
        print(outlier)
    else:
        print('There is no outlier.')

    # construct mask
    count_pair = 0
    count_sample = 0.0
    mask = np.zeros((len(sample_patient), len(sample_patient)))
    for i in range(0, len(df_pair)):
        dfi = df_pair[i]
        nc = dfi[dfi['label'] == 0.0]
        ad = dfi[dfi['label'] == 2.0]
        print('The %d-th pair, \tsamples:%d, \tnc: %d, \tad: %d' % (i, len(dfi), len(nc), len(ad)))
        count_pair += len(nc) * len(ad)
        count_sample += len(dfi)
        k1, k2 = [list(x.index) for x in [nc, ad]]
        assert max(k1) < 606
        assert min(k2) >= 606
        assert (dfi.loc[k1, 'barcodes'] == sample_patient.loc[k1, 'barcodes']).all()
        for j1 in k1:
            for j2 in k2:
                mask[j1, j2] = 1.0
                mask[j2, j1] = 1.0

    print('Finally, %d patients, %d samples, %d pairs' % (len(df_pair), count_sample, count_pair * 2))
    # Finally, 9 patients, 27 samples, 46 pairs
    assert count_pair * 2 == np.sum(mask)
    df_pair = pd.concat(df_pair, axis=0)

    return df_pair, mask


def chr_loc(raw_dir, out_dir):
    """
    chromosome and physical positions of all probes on the chip
    :return:
        HumanMethylation850_probe_chr_loc_sort.csv
    """
    if os.path.isfile(out_dir):
        print('exist chr_loc!')
        return
    df = pd.read_csv(raw_dir, header=7)
    use_col = ['IlmnID', 'Infinium_Design_Type', 'CHR', 'MAPINFO', 'Chromosome_36', 'Coordinate_36', 'Strand',
               'UCSC_RefGene_Name', 'UCSC_RefGene_Accession', 'UCSC_RefGene_Group', 'UCSC_CpG_Islands_Name',
               'Relation_to_UCSC_CpG_Island']
    df_use = df[use_col]
    df_use['CHR'] = [(int(float(x)) if ((x not in ['X', 'Y', np.nan]) and (x == x)) else x) for x in df_use['CHR']]
    df_sort = df_use.sort_values(['CHR', 'MAPINFO'], inplace=False).reset_index(drop=True, inplace=False)
    df_sort.to_csv(out_dir, index=False)
    print('finish saving chr_loc!')


if __name__ == '__main__':
    main()
