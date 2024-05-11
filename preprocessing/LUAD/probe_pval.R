library(data.table)

# --- load data ---
outdir = './../../data/LUAD/part_data/'
df = fread(paste0(outdir, 'all_dropna_chr_loc.csv'))
df = data.frame(df)
print(dim(df))  # 373615    678
emp = 2  # source, label

# --- calculate beta-difference ---
df_tcga = df[, 1:(492+3)]
label = df[emp, 1:(492+3)]
all.n = df_tcga[, which(label==1)]  # dim: 373615     32
all.c = df_tcga[, which(label==0)]  # dim: 373615    460

# --- t-test ---
pvals = rep(NA, nrow(df))
for (i in (emp+1):nrow(df)) {
# if (var(as.numeric(all.n[i,]))==0 & var(as.numeric(all.c[i,]))==0){
 if (var(as.numeric(all.n[i,]))==0 | var(as.numeric(all.c[i,]))==0){
  pvals[i] = 1
}else{
  pvals[i] = t.test(as.numeric(all.n[i,]), as.numeric(all.c[i,]))$p.value
}
}
pvals[c(1:emp)]=rep(0,emp)

# --- p-correction ---
padj = p.adjust(pvals[(emp+1):nrow(df)],method = "BH")
padj = c(rep(0,emp),padj)

# --- save ---
pall = t(rbind(pvals, padj))
write.csv(pall, file = paste0(outdir, "/pvalues.csv"))

pall=read.csv(paste0(outdir, "/pvalues.csv"))
pvals=pall[, 2]
padj=pall[, 3]

thi = 0.005
res = df[which(pvals<thi), ]  # 207884
fwrite(res,file = paste0(outdir, "/pval",thi,".csv"))
