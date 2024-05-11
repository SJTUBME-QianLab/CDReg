library(data.table)

Args <- commandArgs()
cancer.name = Args[3]
print(cancer.name)
datadir = './../raw_data/LUAD/'
predir = './../data/LUAD/'
#install.packages('stringr')
#install.packages('dplyr')
#install.packages('RPMM')
library(stringr)
info = fread(paste0(predir,'HumanMethylation450_probe_chr_loc_sort.csv'))
p2cg = info[,c(1,3)]
type = info[,c(1,2)]
colnames(type) = c('ID', 'type')

meth.table=fread(paste0(datadir,cancer.name), sep = "\t")

# delete sex chromosome
meth.table = merge(p2cg, meth.table, 
                   by.x=colnames(p2cg)[1], by.y=colnames(meth.table)[1])
chrs <- array(meth.table$CHR)
chrxyid <- array(which(chrs == "X" | chrs == "Y"))
if (length(chrxyid) != 0) {
    meth.table <- meth.table[-chrxyid, -2]
}else{
    meth.table <- meth.table[, -2]
}
colnames(meth.table)[1] = 'ID_REF'

table.na <- is.na(meth.table)

# Calculate the proportion of NA in samples, and delete samples with NA content over 30%.
na.in.sample <- array(colSums(table.na) / nrow(table.na))
over.sample <- array(which(na.in.sample > 0.3))
if (length(over.sample) != 0) {
    meth.table <- meth.table[, -over.sample]
}

# Calculate the proportion of NA in probes, and delete probes with NA content over 30%.
na.in.probe <- array(rowSums(table.na) / ncol(table.na))
over.probe <- array(which(na.in.probe > 0.3))
if (length(over.probe) != 0) {
    meth.table <- meth.table[-over.probe, ]
}

#------- Use LSimpute.jar to impute missing values -------
nadata <- data.frame(meth.table)
# only normal
colarray = strsplit(colnames(nadata), '.', fixed = TRUE)
label={1}
for (i in 2:length(colarray)) { label=cbind(label,floor(as.integer(colarray[[i]][4])/10)) }
normaldata <- data.table(nadata[, c(which(label==1))])
fwrite(
    normaldata,
    file = paste("meth" , cancer.name, "txt", sep = "."),
    col.names = T,
    row.names = F,
    sep = "\t"
)

javacmd<-paste("java -jar -server LSimpute.jar meth" , cancer.name, "txt LSmeth" , cancer.name, "pute.txt 2",sep = ".")
system(javacmd)
meth.pute <-fread(paste("LSmeth" , cancer.name, "pute.txt", sep = "."))

#-- Add Probe I OR II --
library("dplyr")
colnames(meth.pute)[1] <- "ID"
#tcga.table[,2]<-as.character(tcga.table[,2])
data.before.norm = merge(meth.pute, type, by = "ID")

if (length(which(data.before.norm$type==1))
    & length(which(data.before.norm$type==2))){
# merge can do the same job as 'inner_join'
#m <- merge(meth.pute,tcga.table, by.x = "ID", by.y = "ID")
fwrite(#meth.ov.before.norm.txt
    data.before.norm,
    file = paste("meth", cancer.name, "before.norm.txt", sep = "."),
    row.names = F,
    col.names = T,
    sep = "\t"
)

#---------- Use BMIQ to correct probes ---------------------
# Rscript BMIQ_1.3.R before_normalized.txt after_normalized.txt
library('cluster')
library('RPMM')
rcmd<-paste("Rscript BMIQ_1.3.R meth",cancer.name,"before.norm.txt BMIQmeth",cancer.name,"pute.norm.txt",sep = ".")
system(rcmd)

alldata=fread(paste('BMIQmeth',cancer.name,'pute.norm.txt',sep='.'))
alldata=data.frame(alldata)
file.remove(paste("meth", cancer.name, "before.norm.txt", sep = "."))
file.remove(paste('BMIQmeth',cancer.name,'pute.norm.txt',sep='.'))

}else{
    alldata = data.frame(meth.pute)
}
    
#save only normal data
normaldata <- data.table(alldata)

label.normal <- cbind('label',t(rep(1,ncol(normaldata)-1)))
source.normal <- cbind('source', t(rep(cancer.name,ncol(normaldata)-1)))
sl.normal <- rbind(source.normal, label.normal)
colnames(sl.normal) <- colnames(normaldata)
data.normal <- rbind(sl.normal, normaldata)
fwrite(
    data.normal,
    file = paste0(predir, '/pre_result/', cancer.name, '.normal.txt'),
    row.names = F, col.names = T, sep = "\t"
)

file.remove(paste("meth", cancer.name, "txt", sep = "."))
file.remove(paste("LSmeth", cancer.name, "pute.txt", sep = "."))
# file.remove(paste("meth", cancer.name, "before.norm.txt", sep = "."))
# file.remove(paste('BMIQmeth',cancer.name,'pute.norm.txt',sep='.'))
