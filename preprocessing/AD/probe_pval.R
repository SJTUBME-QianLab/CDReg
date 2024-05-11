# AD VS NC
library(data.table)

data.dir = './../../data/AD/merge/'
out.dir = './../../data/AD/pval_data/'
print(data.dir)
print(out.dir)

df = fread(paste(data.dir, "ADNI_all_meth_label.txt", sep='/'))
head(df)
df=data.frame(df)
label=df[1,]

# -- calculate beta-difference -----
all.nc.num<-df[,which(label==0)]
all.mci.num<-df[,which(label==1)]
all.ad.num<-df[,which(label==2)]

if (!file.exists(paste0(out.dir, "/pvals.csv"))){
  # ------ t-test --------
  # AD VS NC
  pvals.ad.nc <- rep(NA, nrow(all.nc.num))
  for(i in 2:nrow(all.nc.num)) pvals.ad.nc[i] <- t.test(all.ad.num[i,],all.nc.num[i,])$p.value
  pvals.ad.nc[1]=0

  # AD VS MCI
  pvals.ad.mci <- rep(NA, nrow(all.nc.num))
  for(i in 2:nrow(all.nc.num)) pvals.ad.mci[i] <- t.test(all.ad.num[i,],all.mci.num[i,])$p.value
  pvals.ad.mci[1]=0

  # NC VS MCI
  pvals.mci.nc <- rep(NA, nrow(all.nc.num))
  for(i in 2:nrow(all.nc.num)) pvals.mci.nc[i] <- t.test(all.nc.num[i,],all.mci.num[i,])$p.value
  pvals.mci.nc[1]=0

  pvals = matrix(0, nrow(all.nc.num), 3)
  pvals[, 1] = pvals.ad.nc
  pvals[, 2] = pvals.ad.mci
  pvals[, 3] = pvals.mci.nc
  colnames(pvals) = c('AD-NC', 'AD-MCI', 'MCI-NC')
  write.csv(pvals, file = paste0(out.dir, "/pvals.csv"))
}


probe = df[,1]
# c(dim(all.nc.num),dim(all.mci.num),dim(all.ad.num))
# [1] 693965    606 693965    890 693965    395
nc_ad = cbind(probe,all.nc.num,all.ad.num)  # 1001
pvals = fread(paste0(out.dir, '/pvals.csv'))  # 693965, 4
# pvals[1:3,]
#    V1     AD-NC    AD-MCI    MCI-NC
# 1:  1 0.0000000 0.0000000 0.0000000
# 2:  2 0.5629259 0.7528426 0.7187154
# 3:  3 0.2662803 0.4470819 0.5938459
pvals.ad.nc = pvals$'AD-NC'
pvals.ad.mci = pvals$'AD-MCI'
pvals.mci.nc = pvals$'MCI-NC'

thrd=0.05
nc_ad_p = nc_ad[which(pvals.ad.nc < thrd), ]
fwrite(nc_ad_p,file = paste0(out.dir, "/nc_ad_p",thrd,"_",dim(nc_ad_p)[1]-1,".csv"))

