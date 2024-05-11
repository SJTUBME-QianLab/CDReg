# Title     : TODO
# Objective : TODO
# Created by: admin
# Created on: 2021/11/19

# # R 4.0.3
# options(BioC_mirror="http://mirrors.ustc.edu.cn/bioc/")
# if (!require("BiocManager", quietly = TRUE))  # 1.30.16
#     install.packages('BiocManager', repos='http://cran.us.r-project.org')
# BiocManager::install("ChAMP")  # 2.20.1

library(ChAMP)
library(data.table)

root_dir = "./../../dara/AD/"

dir.create(paste0(root_dir, "/CHAMP/"))
for (i in 1:20){
  diri=paste0(root_dir, "/split/ADNI_iDAT_part", i,'/')
  myLoad <- champ.load(directory =diri,arraytype = "EPIC")
  myNorm <-champ.norm(arraytype="EPIC", core=8)
  df=myNorm
  colnames(df)=myLoad$pd$barcodes
  df = data.frame(df)
  filename=paste0(root_dir, "/CHAMP/ADNI_part", i,'.txt')
  print(filename)
  fwrite(df,filename,row.names=TRUE,sep='\t')
  print('Finish!')
}
