import pandas as pd
import numpy as np
import os


def main():
    out_dir = './../../data/LUAD/'
    raw_dir = './../../raw_data/LUAD/'
    chr_loc(os.path.join(raw_dir, 'humanmethylation450_15017482_v1-2.csv'),
            os.path.join(out_dir, 'HumanMethylation450_probe_chr_loc_sort.csv'))  # get loc.csv file


def chr_loc(raw_dir, out_dir):
    """
    chromosome and physical positions of all probes on the chip
    :return:
        HumanMethylation450_probe_chr_loc_sort.csv
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
