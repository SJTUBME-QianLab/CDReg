# CDReg

This repository holds the code for the paper

**Causality-driven candidate identification for reliable DNA methylation biomarker discovery**

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserves the copyright and all legal rights of these codes.

# Author List

Xinlu Tang, Rui Guo, Zhanfeng Mo, Wenli Fu, and Xiaohua Qian

# Abstract

Despite vast data support in DNA methylation (DNAm) biomarker discovery to facilitate health-care research, this field faces huge resource barriers due to preliminary unreliable candidates and the consequent compensations using expensive experiments. The underlying challenges lie in the confounding factors, especially measurement noise and individual characteristics. To achieve reliable identification of a candidate pool for DNAm biomarker discovery, we propose a Causality-driven Deep Regularization (CDReg) framework to reinforce correlations that are suggestive of causality with disease. It integrates causal thinking, deep learning, and biological priors to handle non-causal confounding factors, through a contrastive scheme and a spatial-relation regularization that reduces interferences from individual characteristics and noises, respectively. The comprehensive reliability of CDReg was verified by simulations and applications involving various human diseases, sample origins, and sequencing technologies, highlighting its universal biomedical significance. Overall, this study offers a causal-deep-learning-based perspective with a compatible tool to identify reliable DNAm biomarker candidates, promoting resource-efficient biomarker discovery.

# System requirements and installation

## Hardware requirements

Our `CDReg` method requires only a standard computer with enough RAM to support the in-memory operations.

## OS Requirements

Our code has been tested on the following systems:

- Windows 10
- Linux: Ubuntu 18.04

## Dependencies

Our code is mainly based on **Python 3.8.19** and **PyTorch 1.10.1**.

Other useful Python libraries:

- NumPy

- pandas

- scikit-learn

- SciPy

## Environment

The environment can be created via conda as follows:

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

## PC

The PC data profiled by WGBS was obtained from MethBank (https://ngdc.cncb.ac.cn/methbank/) with the Project ID HRA000099.

- Download files using wget: `./preprocessing/PC/download.sh`

- Download sample information from https://ngdc.cncb.ac.cn/methbank/samples?q=HRA000099

## Chip annotation

Download the annotation file of DNA methylation microarray chips from Illumina.

- MethylationEPIC
  
  - Access the webpage: https://support.illumina.com/downloads/infinium-methylationepic-v1-0-product-files.html
  
  - Click on the hyperlink `Infinium MethylationEPIC v1.0 B4 Manifest File (CSV Format)`. The downloaded file is named as `infinium-methylationepic-v-1-0-b4-manifest-file-csv.zip`  and can be unzipped to get  `HumanMethylation450MethylationEPIC_v-1-0_B4.csv`
  
  - Copy it to `./raw_data/AD/`

- HumanMethylation450
  
  - Access the webpage: https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
  
  - Click on the hyperlink `HumanMethylation450 v1.2 Manifest File (CSV Format)`. The downloaded file is named as `humanmethylation450_15017482_v1-2.csv` 
  
  - Copy it to `./raw_data/LUAD/`

## Gencode

- Download genecode V27 from http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/

- Run script `./preprocessing/PC/download.sh` to obtain `allGene.hg38.position`

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

## PC

```bash
python PC_pre.py
```

# Running

## Simulation

```shell
cd ./CDReg
python main_simu.py \
    --seed 2020 --data_name covLap2_de1_b0.5_seed2027 \
    --L1 0.4 --L21 0.05 --Ls 1.2 --Lc 0.1 --lr 0.0001
```

Expected output: `info.csv`, `eval_FS1.xlsx`.

## LUAD

```bash
cd ./CDReg
python main_app.py \
    --data_name LUAD \
    --L1 0.5 --L21 0.2 --Ls 1.2 --Lc 0.3 --lr 0.001
```

Expected output: `info.csv`, `final_50_eval1.xlsx`, `eval_FS1.xlsx`, `pred_epoch.csv`, `fea_scores.csv`

## AD

```bash
cd ./CDReg
python main_app.py \
    --data_name AD --batch_size 501 \
    --L1 0.15 --L21 0.04 --Ls 0.5 --Lc 1.2 --lr 0.001
```

## PC

```bash
cd ./CDReg
python main_app.py \
    --data_name CHR20p0.05gp20 \
    --L1 0.5 --L21 0.1 --Ls 1.2 --Lc 1 --lr 0.0001
```

## Classification

```bash
python post_test.py --data_name AD
python post_test.py --data_name LUAD
python post_test.py --data_name CHR20p0.05gp20
```

# Comparing methods

## Requirements

Comparing methods are implemented using R version 4.0.4 with the following required packages:

- glmnet 4.1.3

- SGL 1.3

- pclogit 0.1

- minfi 1.36.0

- DMRcate 2.4.1

## Run commands (examples):

```bash
cd ./comparing
R --no-save < comparing_app.R LUAD
R --no-save < other_app.R LUAD
python eval_R_app.py --data_name LUAD
python post_test.py --data_name LUAD

R --no-save < comparing_simu.R covLap2_de1_b0.5_seed2027 1 LASSO
python eval_R.py --data_name covLap2_de1_b0.5_seed2027 --seed 1 --method LASSO
```

# Reproducibility

`cd ./reproducibility`

- Generate simulation data: `batch_generate_simulation.py`

- Simulation experiments: `batch_simu_ours.py`

# Figures

- Fig. 3-5 and Supplementary Figs. S1-S5 can be generated using the codes in `./plot/`

# Contact

For any questions, feel free to contact

> Xinlu Tang : [tangxl20@sjtu.edu.cn](mailto:tangxl20@sjtu.edu.cn)
