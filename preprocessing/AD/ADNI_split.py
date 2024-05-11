import pandas as pd
import numpy as np
import os
import time
import shutil

raw_data_dir = "./../../raw_data/AD/"
out_data_dir = "./../../data/AD/split"


# Anno
if not os.path.exists(raw_data_dir + "ADNI_DNA_Methylation_SampleAnnotation_20170530.csv"):
    df = pd.read_excel(raw_data_dir + "ADNI_DNA_Methylation_SampleAnnotation_20170530.xlsx", sheet_name='SampleAnnotation')
    for i in range(len(df)):
        df.loc[i, 'Edate'] = str(df.loc[i, 'Edate']).split(' ')[0]
        df.loc[i, 'DateDrawn'] = str(df.loc[i, 'DateDrawn']).split(' ')[0]
    df.rename(columns={'Phase': 'Sample_Group', 'Array': 'Sentrix_Position', 'Slide': 'Sentrix_ID'}, inplace=True)
    df.to_csv(raw_data_dir + "ADNI_DNA_Methylation_SampleAnnotation_20170530.csv", index=False)

# split
dir1 = raw_data_dir + "ADNI_iDAT_files/"
diri = out_data_dir + "ADNI_iDAT_part"
files = os.listdir(dir1)
files.sort(key=lambda x: (int(x[:12]), x[12:-5]))  # sort !!!!!!!!!
processed = set()
i = 1
dir2 = diri + str(i) + '/'
for fi in files:
    print(fi[:-9])  # e.g. 200223270029_R08C01
    if '.idat' not in fi:
        continue
    if not os.path.exists(dir2):
        os.makedirs(dir2)
    if len(processed) == i * len(files) // 40:
        i += 1
        dir2 = diri + str(i) + '/'
    if fi[:-9] not in processed:
        processed.add(fi[:-9])
        old_name = dir1 + fi[:-9] + '_Grn.idat'
        new_name = dir2 + fi[:-9] + '_Grn.idat'
        shutil.copyfile(old_name, new_name)
        old_name = dir1 + fi[:-9] + '_Red.idat'
        new_name = dir2 + fi[:-9] + '_Red.idat'
        shutil.copyfile(old_name, new_name)
print('Finish!')


# create new Ann csv for split data
for i in range(1, 21):
    df = pd.read_csv(raw_data_dir + "ADNI_DNA_Methylation_SampleAnnotation_20170530.csv")
    dir1 = out_data_dir + "ADNI_iDAT_part"+str(i)+"/"
    print(i, dir1)
    files = os.listdir(dir1)  # 顺序不影响
    sets = []
    for filei in files:
        if '.idat' not in filei:
            continue
        sets.append(filei[:-9])

    for k in range(df.shape[0]):
        if df['barcodes'][k] not in sets:
            df['barcodes'][k] = np.nan

    df.dropna(inplace=True)
    df.to_csv(dir1+'newann.csv', index=False)
