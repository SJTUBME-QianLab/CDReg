import pandas as pd
import numpy as np
import os
import time
import re


def main():
    out_dir = './../../data/LUAD/'
    out_dir0 = '/'
    pre_dir = '/'
    # loc_dir = os.path.join(out_dir, 'HumanMethylation450_probe_chr_loc_sort.csv')
    loc_dir = '/'

    # # 1. extract LUAD data
    # final_d, part_tcga, part_geo = extract_data(loc_dir, pre_dir)
    # # 460 32 (492) 164 19 (183) -> 675
    # os.makedirs(os.path.join(out_dir, 'part_data'), exist_ok=True)
    # final_d.to_csv(os.path.join(out_dir, 'part_data', 'all_dropna_chr_loc.csv'), index=False)
    # part_tcga.to_csv(os.path.join(out_dir, 'part_data', 'part_tcga.csv'), index=False)
    # part_geo.to_csv(os.path.join(out_dir, 'part_data', 'part_geo.csv'), index=False)
    # # shape:  (373615, 678) first 2 rows are source and label, 373615-2=373613 probes

    # # 2. p-value screen
    # os.system('R --no-save < probe_pval.R')
    # # pval < 0.005, shape:(207884, 678), probes: 207882
    # # ###############################################################

    gp_min = 20
    # 3. group by gene
    data = pd.read_csv(os.path.join(out_dir, 'part_data', 'pval0.005.csv'))
    info1, final_data = extract_1gene_probes_450k(loc_dir, out_dir, data, gp_min)
    info1.to_csv(os.path.join(out_dir, 'group_gene', 'info.csv'), index=False)
    final_data.to_csv(os.path.join(out_dir, 'group_gene', 'data_1gene_gt%d.csv' % gp_min), index=False)

    # 4. barcode pair
    df_pair, mask = get_pair(data.iloc[:2, 3:(492+3)])
    # Finally, 29 patients, 58 samples, 58 pairs
    df_pair.to_csv(os.path.join(out_dir, 'group_gene', 'pair_barcode.csv'), index=True)
    np.save(os.path.join(out_dir, 'group_gene', 'pair_matrix.npy'), mask)

    # 5. normalize
    final_data = pd.read_csv(os.path.join(out_dir, 'group_gene', 'data_1gene_gt%d.csv' % gp_min))
    X = final_data.iloc[1:, 3:].values.T.astype(float)  # sample * feature
    nn = np.linalg.norm(X, ord=2, axis=0)
    X_norm = X / nn[None, :]  # n*d / 1*d
    np.save(os.path.join(out_dir, 'group_gene', 'X_normL2.npy'), X_norm)
    Y = final_data.iloc[0, 3:].values.astype(int)
    np.save(os.path.join(out_dir, 'group_gene', 'Y.npy'), Y)


def extract_data(loc_dir, pre_dir):
    col = 'ID'
    train_c = pd.read_csv(os.path.join(pre_dir, 'xena', 'LUAD.cancer.txt'), sep='\t', low_memory=False)
    train_n = pd.read_csv(os.path.join(pre_dir, 'xena', 'LUAD.normal.txt'), sep='\t', low_memory=False)
    test_c = pd.read_csv(os.path.join(pre_dir, 'geo', 'GSE66836.cancer.txt'), sep='\t', low_memory=False)
    test_n = pd.read_csv(os.path.join(pre_dir, 'geo', 'GSE66836.normal.txt'), sep='\t', low_memory=False)

    all_data = pd.merge(train_c, train_n, on=col)
    all_data = pd.merge(all_data, test_c, on=col)
    all_data = pd.merge(all_data, test_n, on=col)
    all_data_dropna = all_data.dropna(axis=0)
    length = [train_c.shape[1] - 1, train_n.shape[1] - 1, test_c.shape[1] - 1, test_n.shape[1] - 1]

    print('after dropna, shape: ', all_data_dropna.shape)

    # 加入染色体坐标
    loc = pd.read_csv(loc_dir)
    loc = loc[['IlmnID', 'CHR', 'MAPINFO']]
    loc.loc[-2] = ['source', 'source', 'source']
    loc.loc[-1] = ['label', 'label', 'label']
    loc.index = loc.index + 2
    loc = loc.sort_index()

    final = pd.merge(loc, all_data_dropna, left_on='IlmnID', right_on=col, how='inner').drop([col], axis=1)
    final_d = final.dropna(subset=['CHR', 'MAPINFO'])
    print('shape: ', final_d.shape)

    # partion
    tcga_len = length[0] + length[1]
    geo_len = length[2] + length[3]
    part_tcga = final_d.iloc[:, :tcga_len + 3]
    part_geo = final_d.iloc[:, [0, 1, 2] + list(range(tcga_len + 3, tcga_len + geo_len + 3))]

    return final_d, part_tcga, part_geo


def extract_1gene_probes_450k(loc_dir, out_dir, data, gp_min=5):
    assert list(data.columns[:3]) == ['IlmnID', 'CHR', 'MAPINFO']
    assert data.iloc[1, 0] == 'label' and data.iloc[0, 0] == 'source'
    data1 = data.drop(0, axis=0, inplace=False).reset_index(drop=True, inplace=False)
    data1.iloc[0, :3] = ['-1', -1, -1]
    data1.iloc[:, [1, 2]] = data1.iloc[:, [1, 2]].astype(float).astype(int)
    data1.iloc[0, 3:] = data1.iloc[0, 3:].astype(int)
    data1.iloc[1:, 3:] = data1.iloc[1:, 3:].astype(float)

    if os.path.isfile(os.path.join(out_dir, 'group_gene', 'allprobe_gene.csv')):
        print('exist allprobe_gene.csv')
        pr_gene = pd.read_csv(os.path.join(out_dir, 'group_gene', 'allprobe_gene.csv'))
    else:
        data1_use = data1.iloc[:, :3]
        
        # add chromosome (CHR) and location (MAPINFO)
        loc = pd.read_csv(loc_dir)
        loc = loc[['IlmnID', 'UCSC_RefGene_Name']]
        loc.loc[-1] = ['-1', '-1']
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
                # print(i)
                gene_set = sorted(set(pr_gene.loc[i, 'UCSC_RefGene_Name'].split(';')))
                pr_gene.loc[i, 'gene_set'] = ';'.join(gene_set)
                pr_gene.loc[i, 'gene_num'] = len(gene_set)

        pr_gene.iloc[0, 1] = '-1'
        pr_gene.iloc[0, 2] = 1
        os.makedirs(os.path.join(out_dir, 'group_gene'), exist_ok=True)
        pr_gene.to_csv(os.path.join(out_dir, 'group_gene', 'allprobe_gene.csv'), index=False)
        print('save allprobe_gene.csv!')

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

        drop_gp_idx = info1[info1['gene_set'] == drop_gene[0]]['gp_idx'].values
        info2['gp_idx'] = [-1] + [(x - 1 if x > drop_gp_idx[0] else x) for x in info2['gp_idx'][1:]]

        assert max(info2['gp_idx']) + 2 == len(group.dropna(inplace=False))
        info1 = info2.copy()
        del info2

    for i in range(2, len(info1)):
        if info1.loc[i, 'CHR'] < info1.loc[i - 1, 'CHR']:
            print('wrong: %d' % i)
        if info1.loc[i, 'CHR'] == info1.loc[i - 1, 'CHR'] and info1.loc[i, 'MAPINFO'] < info1.loc[i - 1, 'MAPINFO']:
            print('wrong: %d' % i)

    final_data = pd.merge(info1[['IlmnID']], data1, on='IlmnID')

    # gp_idx_list = [np.where(info1['gp_idx'] == xx)[0] - 1 for xx in np.unique(info1['gp_idx'][1:])]
    # assert (np.concatenate(gp_idx_list) == list(range(len(info1)-1))).all()

    return info1, final_data


def get_pair(data):
    sample_name = list(data.columns)
    patient = [str(x.split('.')[2]) for x in sample_name]
    sample_patient = pd.DataFrame({
        'barcode': sample_name,
        'patient': patient,
        'label': list(data.iloc[1, ]),
        # 'source': list(data.iloc[0, 1:]),
        # 'label': list(data.iloc[1, 1:]),
    })
    df_dup = sample_patient[sample_patient['patient'].duplicated(keep=False)]
    
    patient_list = sorted(list(set(df_dup['patient'])))
    outlier = []
    df_pair = []
    for pp in patient_list:
        ll = df_dup[df_dup['patient'] == pp]
        if set(ll['label']) == {'0', '1'}:
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

    df_pair = pd.concat(df_pair, axis=0)
    print(df_pair.shape)
    assert sum(df_pair['label'] == '0') == sum(df_pair['label'] == '1')

    # construct mask
    count_pair = 0
    count_sample = 0.0
    mask = np.zeros((len(sample_patient), len(sample_patient)))
    for i in range(0, len(df_pair), 2):
        k1, k2 = df_pair.index[i:(i + 2)]
        assert set(df_pair.loc[[k1, k2], 'label']) == {'0', '1'}
        assert df_pair.loc[k1, 'barcode'] == sample_name[k1]
        count_pair += 1
        count_sample += 2
        mask[k1, k2] = 1.0
        mask[k2, k1] = 1.0

    print('Finally, %d patients, %d samples, %d pairs' % (len(df_pair)/2, count_sample, count_pair * 2))
    assert count_pair * 2 == np.sum(mask)
    # Finally, 29 patients, 58 samples, 58 pairs

    return df_pair, mask


def tmp():
    old_dir = '/home/data/tangxl/ContrastSGL/casecontrol_data/LUAD/group_gene/'
    new_dir = '/home/tangxl/ContrastSGL/20240510-clean/data/LUAD/group_gene/'
    for ff in os.listdir(new_dir):
        print(ff)
        if os.path.isdir(os.path.join(new_dir, ff)):
            continue
        assert os.path.isfile(os.path.join(old_dir, ff))
        if ff == 'allprobe_gene_850k_p0.05.csv':
            continue
        if ff.endswith('.csv'):
            # if ff == 'basic_info.csv':
            # new = pd.read_csv(os.path.join(new_dir, ff))
            # old = pd.read_csv(os.path.join(old_dir, ff))
            # pd.testing.assert_frame_equal(new, old)
            # else:
            # new = pd.read_csv(os.path.join(new_dir, ff), header=None)
            # old = pd.read_csv(os.path.join(old_dir, ff), header=None)
            # np.allclose(new.values, old.values, atol=1e-5, equal_nan=True)
            new = pd.read_csv(os.path.join(new_dir, ff))
            old = pd.read_csv(os.path.join(old_dir, ff))
            try:
                pd.testing.assert_frame_equal(new.fillna('nan'), old.fillna('nan'))
            except Exception as e:
                print(e)
        elif ff.endswith('.npy'):
            new = np.load(os.path.join(new_dir, ff))
            old = np.load(os.path.join(old_dir, ff))
            if not (new == old or np.allclose(new, old, atol=1e-5)):
                print('not equal')
        elif ff.endswith('.npz'):
            from scipy import sparse
            new = sparse.load_npz(os.path.join(new_dir, ff))
            old = sparse.load_npz(os.path.join(old_dir, ff))
            assert (new.data == old.data).all() or np.allclose(new.data, old.data, atol=1e-8)
            # old = pd.read_csv('/home/data/tangxl/ContrastSGL/casecontrol_data/LUAD/HumanMethylation450_probe_chr_loc_sort.csv')
            # new = pd.read_csv('/home/tangxl/ContrastSGL/20240510-clean/data/LUAD/HumanMethylation450_probe_chr_loc_sort.csv')
            # pd.testing.assert_frame_equal(old.fillna('nan'), new.fillna('nan'))


if __name__ == '__main__':
    main()