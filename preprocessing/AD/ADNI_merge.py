import pandas as pd
import numpy as np
import os
import time

raw_dir = './../../raw_data/AD/'
folder_path = "./../../data/AD/"
os.makedirs(folder_path, exist_ok=True)


def main():
    Label_convert()
    chr_loc()

    Merge_data()
    Insert_label()


def Merge_data():
    """
    merge 20 files after ChAMP
    :return: ADNI_all_meth.txt
    """
    df = pd.read_csv(folder_path+"CHAMP/ADNI_part1.txt", sep='\t')
    df.rename(columns={df.columns[0]: 'probe'}, inplace=True)

    for numi in range(2, 21):
        filei = "ADNI_part"+str(numi)+".txt"
        dfi = pd.read_csv(folder_path+"CHAMP/"+filei, sep='\t')
        dfi.rename(columns={dfi.columns[0]: 'probei'}, inplace=True)
        df = df.merge(dfi, left_on='probe', right_on='probei', suffixes=(False, False))
        del df['probei']
        print('Finish merge ', str(numi), '!')
    df.to_csv(folder_path+'merge/ADNI_all_meth.txt', sep='\t', index=False)
    print('Finish merge all parts!')


def Label_convert():
    """
    Sample labels
    :return: ADNI_meth_barcodes2label.csv
    """
    if os.path.isfile(os.path.join(folder_path, 'merge/ADNI_meth_barcodes2label.csv')):
        return

    tadpole = pd.read_csv(raw_dir + 'TADPOLE_D1_D2.csv',
                          dtype=np.object)
    adni = pd.read_excel(raw_dir + 'ADNI_DNA_Methylation_SampleAnnotation_20170530.xlsx',
                         sheet_name='SampleAnnotation', dtype=np.object)

    adni['EXAMDATE'] = np.nan
    for i in range(len(adni)):
        adni.loc[i, 'EXAMDATE'] = str(adni.loc[i, 'Edate']).split(' ')[0]

    for i in range(len(tadpole)):
        tadpole.loc[i, 'RID'] = int(tadpole.loc[i, 'RID'])

    merge = pd.merge(adni, tadpole, on=['RID', 'EXAMDATE'])
    use = merge.loc[:, ['barcodes', 'DXCHANGE']]

    # label
    use.reset_index(drop=True, inplace=True)
    DCdict = {'1': 0, '2': 1, '3': 2, '4': 1, '5': 2, '6': 2, '7': 0, '8': 1}
    data = use.dropna(axis=0, how='any')
    data['label_NC0_MCI1_AD2'] = data['DXCHANGE'].apply(lambda n: DCdict[n])

    data_final = data.T
    data_final.columns = data_final.loc['barcodes', :]
    data_final.drop(['barcodes', 'DXCHANGE'], axis=0, inplace=True)
    data_final.to_csv(folder_path + 'merge/ADNI_meth_barcodes2label.csv')


def Insert_label():
    """
    insert ADNI_meth_barcodes2label.csv to ADNI_all_meth.txt
    :return: ADNI_all_meth_label.txt
    """
    rawdf = pd.read_csv(os.path.join(folder_path, "merge/ADNI_all_meth.txt"), sep='\t')
    label_df = pd.read_csv(folder_path + "merge/ADNI_meth_barcodes2label.csv")
    label_df.columns = ["X" + x for x in list(label_df.columns)]
    rawdf.rename(columns={rawdf.columns[0]: 'probe'}, inplace=True)
    label_df.rename(columns={label_df.columns[0]: 'probe'}, inplace=True)

    df = pd.concat([label_df, rawdf], join='inner')
    df.to_csv(os.path.join(folder_path, "merge/ADNI_all_meth_label.txt"), sep='\t', index=False)


if __name__ == '__main__':
    main()
