# CDReg

This repository holds the code for the paper

**Causal-aware candidate identification for resource-efficient DNA methylation biomarker discovery**

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserves the copyright and all legal rights of these codes.

# Author List

Xinlu Tang, Rui Guo, Zhanfeng Mo, Wenli Fu, and Xiaohua Qian

# Abstract

Despite vast data support in DNA methylation (DNAm) biomarker discovery to facilitate health-care research, this field faces huge resource barriers due to preliminary unreliable candidates and the consequent compensations using expensive experiments. The underlying challenges lie in the confounding factors, especially measurement noise and individual characteristics. To achieve reliable identification of a candidate pool for DNAm biomarker discovery, we propose a causal-aware deep regularization (CDReg) framework. It innovatively integrates causality, deep learning, and biological priors to handle non-causal confounding factors, through a contrastive scheme and a spatial-relation regularization that reduces interferences from individual characteristics and noises, respectively. The comprehensive reliability, attributed to the specific challenge-solving ability, of CDReg was verified by simulations, applications on cancer tissue, and explorations for blood samples on neurological disease, highlighting its biomedical significance. Overall, this study offers a novel causal-deep-learning-based perspective with a compatible tool for reliable candidate identification to achieve resource-efficient DNAm biomarker discovery.

# Requirements

Our code is mainly based on **Python 3.8.12** and **PyTorch 1.10.1**. The corresponding environment may be created via conda as follows:

```shell
conda env create -f ./environment.yaml
conda activate environment
```

# Raw Data

## LUAD-TCGA

The LUAD dataset is obtained from TCGA through the UCSC Xena platform.

- Access the website: https://xenabrowser.net/datapages/?hub=https://tcga.xenahubs.net:443

- Click on the following hyperlinks: `TCGA Lung Adenocarcinoma (LUAD)` --> `Methylation450k (n=492) TCGA Hub`

- Get the link after "download" to download raw data named `TCGA.LUAD.sampleMap_HumanMethylation450.gz`

## LUAD-GEO-GSE66836

The dataset used for independent testing is obtained from GEO accession numbered GSE66836.

- Access the website: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66836

- Click on the hyperlink `Series Matrix File(s)` to download raw data named `GSE66836_series_matrix.txt.gz`.

## AD

The AD dataset is obtained from the ADNI database: http://adni.loni.usc.edu . Access requires registration and requestion, which includes institutional support and justification of data use.

- Log in ADNI database

- Access the webpage: https://ida.loni.usc.edu/pages/access/geneticData.jsp

- Click on the hyperlink `DNA methylation profiling` and `ALL`.

- Download the following two files:
  
  - Whole-genome DNA methylation profiling Data: `ADNI_DNA_Methylation_iDAT_files.tar.gz`
  
  - Whole-genome DNA methylation profiling Annotations: `ADNI_DNA_Methylation_SampleAnnotation_20170530.xlsx`

- Unzip `ADNI_DNA_Methylation_iDAT_files.tar.gz` to the package `ADNI_iDAT_files`

- Access the webpage: [IDA](https://ida.loni.usc.edu/pages/access/studyData.jsp?categoryId=43&subCategoryId=94) (to obtain class labels of these DNAm data samples)

- Click on the hyperlink `Tadpole Challenge Data` to download a file named `tadpole_challenge_201911210.zip`

- Unzip it and copy the file named `TADPOLE_D1_D2.csv` to `./raw_data/TADPOLE`

## Chip annotation

Download the annotation file of DNA methylation microarray chips from Illumina.

- MethylationEPIC
  
  - Access the webpage: https://support.illumina.com/downloads/infinium-methylationepic-v1-0-product-files.html
  
  - Click on the hyperlink `Infinium MethylationEPIC v1.0 B4 Manifest File (CSV Format)`. The downloaded file is named as `infinium-methylationepic-v-1-0-b4-manifest-file-csv.zip`  and can be unzipped to get  `HumanMethylation450MethylationEPIC_v-1-0_B4.csv`
  
  - Copy it to `./raw_data/AD/`
  
  - Access the webpage: https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
  
  - Click on the hyperlink `HumanMethylation450 v1.2 Manifest File (CSV Format)`. The downloaded file is named as `humanmethylation450_15017482_v1-2.csv` 
  
  - Copy it to `./raw_data/LUAD/`

# Data processing

## Simulation

```shell
python simulate_individual.py --cov Laplacian --rho 2 --delta 1 --bias 0.5 --seed 2027
```

## LUAD

```shell
python get_annotation.py
R --no-save < LSimpute-predata-xena-Arg-cancer.R LUAD
R --no-save < LSimpute-predata-xena-Arg-normal.R LUAD
R --no-save < LSimpute-predata-GEO-Arg.R GSE66836
python LUAD_pre.py
```

## AD

Required R package: ChAMP, data.table

Copy `ADNI_DNA_Methylation_SampleAnnotation_20170530.csv` to the package `ADNI_iDAT_files`

```bash
python ADNI_split.py
R --no-save < batch_CHAMP.R
python ADNI_merge.py
R --no-save < probe_pval.R
python AD_pre.py
```

# Running

## Simulation

```shell
cd ./CDReg
python main_simu.py \
    --seed 2020 --data_name covLap2_de1_b0.5_seed2027 \
    --L1 0.4 --L21 0.05 --Ls 1.2 --Lc 0.1 --lr 0.0001
```

## LUAD

```bash
cd ./CDReg
python main_app.py \
    --data_name LUAD \
    --L1 0.5 --L21 0.2 --Ls 1.2 --Lc 0.3 --lr 0.001
```

## AD

```bash
cd ./CDReg
python main_app.py \
    --data_name AD --batch_size 501 \
    --L1 0.15 --L21 0.04 --Ls 0.5 --Lc 1.2 --lr 0.001
```

## Classification

```bash
python independent_test.py --data_name AD --test_type resample
python independent_test.py --data_name LUAD --test_type resample
```

# Comparing methods

## Requirements

Comparing methods are implemented using R version 4.0.4 with the following required packages:

- Matrix

- glmnet

- grpreg

- SGL

- pclogit

## Run commands:

```bash
cd ./comparing
R --no-save < comparing_app.R LUAD
python eval_R_app.py --data_name LUAD --set_name 3m_default100_0.0001

R --no-save < comparing_app.R AD
python eval_R_app.py --data_name AD --set_name 3m_default100_0.0001

R --no-save < comparing_simu.R covLap2_de1_b0.5_seed2027 1 LASSO
python eval_R.py --data_name covLap2_de1_b0.5_seed2027 --seed 1 --method LASSO
```

# Reproducibility

`./reproducibility`

- Generate simulation data: `batch_generate_simulation.py`

- Simulation experiments: `batch_simu_ours.py`

# Figures

- Results for figures were saved in `Results_simu.xlsx`, `Results_LUAD.xlsx`, and `Results_AD.xlsx`

- Fig. 3-5, Extended Data Fig. 1, and Supplementary Fig. 1 can be generated by the codes in `./plot/`

# Contact

For any questions, feel free to contact

> Xinlu Tang : [tangxl20@sjtu.edu.cn](mailto:tangxl20@sjtu.edu.cn)
