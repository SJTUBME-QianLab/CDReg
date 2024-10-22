Args <- commandArgs()
data_name = Args[3]
# data_name = 'LUAD'
seed = 2020

# environment: ContrastSGL, R=4.0.4
library("minfi")  # 1.36.0
library("DMRcate")  # 2.4.1
library("DMRcatedata")  # 2.8.2

data_dir_root = "./../data/"
out_dir = paste0("./../results/",data_name,"/other/")
if (!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}
set.seed(seed)

#### load data ####
if (data_name == 'LUAD') {
  data_dir = paste0(data_dir_root, data_name, "/group_gene/")
  X = read.table(paste0(data_dir, "data_1gene_gt20.csv"), sep=",", header=T, row.names=1)
  Y = as.numeric(X[1, c(-1,-2)])
  X = X[-1, c(-1,-2)]
  train_idx = c(1:492)
  X_train = as.matrix(X[, train_idx])
  Y_train = Y[train_idx]
} else if (data_name == 'AD'){
  data_dir = paste0(data_dir_root, data_name, "/group_gene/")
  X = read.table(paste0(data_dir, "data_p0.05_1gene_gt5.csv"), sep=",", header=T, row.names=1)
  Y = as.numeric(X[1, ])
  X = X[-1, ]
  X_train = as.matrix(X)  # feature*sample
  Y_train = Y
} else if (grep('^CHR\\w+', data_name)) {
  data_dir = paste0(data_dir_root, data_name)
  X = read.table(paste0(data_dir, "/matrix_met.csv"), sep=",", header=T, row.names=1)
  Y = read.table(paste0(data_dir, "/Y.csv"), sep=",")
  X_train = as.matrix(X)
  Y_train = as.numeric(as.matrix(Y[,1]))
} else {
  stop("wrong data_name")
}

#### limma / minfi / dmp ####
if (!dir.exists(paste0(out_dir, '/dmp10'))){
  dir.create(paste0(out_dir, '/dmp10'), recursive = TRUE)
}
file.copy(from= 'other_app.R', to=paste0(out_dir, '/dmp10'), overwrite=TRUE, copy.date=TRUE)
fit = dmpFinder(dat=X_train, pheno=Y_train, type="continuous")  # order of rows has changed
# intercept, beta, t, pval, qval
write.csv(fit, file = paste0(out_dir, '/dmp10/coef.csv'), row.names = T)

#### DMRcate ####
if (!dir.exists(paste0(out_dir, '/DMRcate'))){
  dir.create(paste0(out_dir, '/DMRcate'), recursive = TRUE)
}
file.copy(from= 'other_app.R', to=paste0(out_dir, '/DMRcate'), overwrite=TRUE, copy.date=TRUE)
if (data_name == 'LUAD') {
  data = cpg.annotate(
    X_train,
    what = "Beta",
    design = model.matrix(~Y_train),
    datatype = "array",
    analysis.type = "differential",
    arraytype = "450K",
    annotation = c(array = "IlluminaHumanMethylation450k", annotation = "ilmn12.hg19"),
    coef = "Y_train",
  )
  hg = "hg19"
} else if (data_name == 'AD') {
  data = cpg.annotate(
    X_train,
    what = "Beta",
    design = model.matrix(~Y_train),
    datatype = "array",
    analysis.type = "differential",
    arraytype = "EPIC",
    annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"),
    coef = "Y_train",
  )
  hg = "hg19"
} else if (grep('^CHR\\w+', data_name)){
  met = read.table(paste0(data_dir, "/met.csv"), sep=",", header=T, row.names=1)
  cov = read.table(paste0(data_dir, "/cov.csv"), sep=",", header=T, row.names=1)
  met[is.na(met)] = 0
  cov[is.na(cov)] = 0
  if (all(rownames(X_train) == rownames(met)) * all(rownames(X_train) == rownames(cov)) != 1){
    stop("rownames not match")
  }
  if (all(colnames(X_train) == colnames(met)) * all(colnames(X_train) == colnames(cov)) != 1){
    stop("colnames not match")
  }
  chr = sapply(strsplit(rownames(met), ":"), function(x) x[1])
  pos = sapply(strsplit(rownames(met), ":"), function(x) as.numeric(x[2]))
  library('bsseq')
  bs_tmp = BSseq(chr=chr, pos=pos, M=as.matrix(met), Cov=as.matrix(cov), sampleNames=colnames(met))
  data = sequencing.annotate(
    bs_tmp,
    methdesign = edgeR::modelMatrixMeth(model.matrix(~Y_train)),
    all.cov = FALSE,  # CpG sites for which some (not all) samples have coverage=0 will be retained
    coef = "Y_train",
  )
  hg = "hg38"
}
output <- dmrcate(data)
DMR <- as.data.frame(extractRanges(output, genome = hg))
write.csv(DMR, file = paste0(out_dir, '/DMRcate/coef.csv'), row.names = T)
