Args <- commandArgs()
data_name = Args[3]
# fold = Args[4]
# data_name = 'LUAD'
seed = 2020
library("glmnet")
library("grpreg")
library("SGL")
library("pclogit")

data_dir_root = "./../data/"
out_dir = paste0("./../results/",data_name,"/3m_default100_0.0001/")

if (!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}
file.copy(from= 'benchmark_app.R', to=out_dir, overwrite=TRUE)
method = "gaussian"
set.seed(seed)

#### load data ####
if (data_name == 'LUAD' | data_name == 'AD') {
  data_dir = paste0(data_dir_root, data_name, "/group_gene/")
  X = read.table(paste0(data_dir, "X_normL2.csv"), sep=",", header=F)
  Y = read.table(paste0(data_dir, "Y.csv"), sep=",", header=F)
  if (data_name == 'LUAD'){
    train_idx = c(1:492)
    X_train = X[train_idx, ]
    Y_train = Y[train_idx, ]
  } else if (data_name == 'AD'){
    X_train = X
    Y_train = Y
  }
  basic_info = read.table(paste0(data_dir, "info.csv"), sep=",", header=T)
  group = as.numeric(as.matrix(basic_info$gp_idx)[-1])
  fea_name = basic_info$IlmnID[-1]
} else if (grep('^CHR\\w+', data_name)) {
  data_dir = paste0(data_dir_root, data_name)
  X = read.table(paste0(data_dir, "X_normL2.csv"), sep=",")
  Y = read.table(paste0(data_dir, "Y.csv"), sep=",")
  X_train = X
  Y_train = as.matrix(Y)
  basic_info = read.table(paste0(data_dir, "sites.csv"), sep=",", header=T, comment.char="")
  group = as.numeric(as.matrix(basic_info$gp_idx))
  fea_name = basic_info$probe
} else {
  stop("wrong data_name")
}

#### Resample 100 times for Selection Probability ####
p = dim(X_train)[2]
resample_time = 1
resample_rate = 1
fea_prob = matrix(0, p, resample_time)
fea_rank = matrix(0, p, resample_time)

#### Enet 0.8 ####
if (!dir.exists(paste0(out_dir, '/Enet0.8'))){
  dir.create(paste0(out_dir, '/Enet0.8'), recursive = TRUE)
}
cvfit = cv.glmnet(as.matrix(X_train), as.matrix(Y_train), alpha=0.8, family=method)
# lambda_list = cvfit$lambda[c(10,20,30,40,50,60,70,80,90,100)]
# lambda_list = c(cvfit$lambda.1se)
lambda_list = c(cvfit$lambda[length(cvfit$lambda)])  # last one (smallest)
# lambda_list = c(0.0005)
write.csv(cvfit$lambda, file = paste0(out_dir, '/Enet0.8/Enet0.8_lambda_list.csv'), row.names = T)
for (ll in 1:length(lambda_list)){
  lambdai = lambda_list[ll]
  for (i in 1:resample_time){
    resample_idx=sample(1:nrow(X_train), round(nrow(X_train)*resample_rate))  # resample
    X_tmp = X_train[resample_idx, ]
    Y_tmp = Y_train[resample_idx]

    # fit = glmnet(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=0.5)  # Enet. [default: alpha=1, LASSO]
    # coeff = as.matrix(coef(fit, s=fit$lambda.min))
    # cvfit = cv.glmnet(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=0.8, family=method)
    # lambda = cvfit$lambda[which(cvfit$cvm==min(cvfit$cvm))]
    fit = glmnet(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=0.8, lambda=lambdai, family=method)  # Enet. [default: alpha=1, LASSO]
    coeff = as.matrix(coef(fit))
    colnames(coeff) = lambdai
    write.csv(coeff, file = paste0(out_dir, '/Enet0.8/Enet0.8_coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

    fea_prob[, i] = I(coeff[-1, ]!=0)
  }
  fea_prob_sum = rowSums(fea_prob)/resample_time
  fea_prob1 = cbind(fea_prob, fea_prob_sum)
  colnames(fea_prob1)[resample_time+1]="ave"
  write.csv(fea_prob1, file = paste0(out_dir, '/Enet0.8/Enet0.8_prob_lam',ll*10,'.csv'), row.names = T)
}


#### LASSO ####
if (!dir.exists(paste0(out_dir, '/LASSO'))){
  dir.create(paste0(out_dir, '/LASSO'), recursive = TRUE)
}
cvfit = cv.glmnet(as.matrix(X_train), as.matrix(Y_train), alpha=1, family=method)
# lambda_list = cvfit$lambda[c(10,20,30,40,50,60,70,80,90,100)]
# lambda_list = c(cvfit$lambda.1se)
lambda_list = c(cvfit$lambda[length(cvfit$lambda)])  # last one (smallest)
# lambda_list = c(0.0005)
write.csv(cvfit$lambda, file = paste0(out_dir, '/LASSO/LASSO_lambda_list.csv'), row.names = T)
for (ll in 1:length(lambda_list)){
  lambdai = lambda_list[ll]
  for(i in 1:resample_time){
    resample_idx=sample(1:nrow(X_train), round(nrow(X_train)*resample_rate))  # resample
    X_tmp = X_train[resample_idx, ]
    Y_tmp = Y_train[resample_idx]

    # fit = glmnet(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=1)  # [default: alpha=1, LASSO]
    # coeff = as.matrix(coef(fit, s=fit$lambda.min))
    # cvfit = cv.glmnet(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=1, family=method)
    # lambda = cvfit$lambda[which(cvfit$cvm==min(cvfit$cvm))]
    fit = glmnet(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=1, lambda=lambdai, family=method)  # Enet. [default: alpha=1, LASSO]
    coeff = as.matrix(coef(fit))
    colnames(coeff) = lambdai
    write.csv(coeff, file = paste0(out_dir, '/LASSO/LASSO_coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

    fea_prob[, i] = I(coeff[-1, ]!=0)
  }
  fea_prob_sum = rowSums(fea_prob)/resample_time
  fea_prob1 = cbind(fea_prob, fea_prob_sum)
  colnames(fea_prob1)[resample_time+1]="ave"
  write.csv(fea_prob1, file = paste0(out_dir, '/LASSO/LASSO_prob_lam',ll*10,'.csv'), row.names = T)
}


#### SGL sparse group lasso ####
if (!dir.exists(paste0(out_dir, '/SGLASSO'))){
  dir.create(paste0(out_dir, '/SGLASSO'), recursive = TRUE)
}
if (method == "gaussian"){
  type = "linear"
}else{
  return(message("wrong method"))
}
# cvfit = cvSGL(data=list(x=as.matrix(X_train), y=as.matrix(Y_train)),
#               index=group, type=type)
# lambda_list = cvfit$lambda[c(1:10)*2]
# lambda_list = c(cvfit$lambda[length(cvfit$lambda)])  # total 20, last one (smallest)
if (data_name == "LUAD"){
  lambda_list = c(0.0001)
}else{lambda_list = c(0.0001)}
# write.csv(cvfit$lambda, file = paste0(out_dir, '/SGLASSO/SGLASSO_lambda_list.csv'), row.names = T)
for (ll in 1:length(lambda_list)){
  lambdai = lambda_list[ll]
  for (i in 1:resample_time){
    resample_idx=sample(1:nrow(X_train), round(nrow(X_train)*resample_rate))  # resample
    X_tmp = X_train[resample_idx, ]
    Y_tmp = Y_train[resample_idx]

    fit = SGL(data=list(x=as.matrix(X_tmp), y=as.matrix(Y_tmp)),
              index=group, type=type, lambdas=lambdai)
    coeff = as.matrix(fit$beta)
    colnames(coeff) = lambdai
    write.csv(coeff, file = paste0(out_dir, '/SGLASSO/SGLASSO_coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

    fea_prob[, i] = I(coeff!=0)
  }
  fea_prob_sum = rowSums(fea_prob)/resample_time
  fea_prob1 = cbind(fea_prob, fea_prob_sum)
  colnames(fea_prob1)[resample_time+1]="ave"
  write.csv(fea_prob1, file = paste0(out_dir, '/SGLASSO/SGLASSO_prob_lam',ll*10,'.csv'), row.names = T)
}

#### pclogit ####
if (!dir.exists(paste0(out_dir, '/pclogit'))){
  dir.create(paste0(out_dir, '/pclogit'), recursive = TRUE)
}
group_size = rep(0, (max(group)+1))
for (k in 1:(max(group)+1)){
  group_size[k] = length(which(group==k-1))
}
fit = pclogit(as.matrix(X_train), as.matrix(Y_train), alpha=0.8, group=group_size)
lambda_list = c(fit$lambda[length(fit$lambda)])  # last one (smallest)
# lambda_list = c(0.0005)
# write.csv(cvfit$lambda, file = paste0(out_dir, '/Enet0.8/Enet0.8_lambda_list.csv'), row.names = T)
for (ll in 1:length(lambda_list)){
  lambdai = lambda_list[ll]
  for (i in 1:resample_time){
    resample_idx=sample(1:nrow(X_train), round(nrow(X_train)*resample_rate))  # resample
    X_tmp = X_train[resample_idx, ]
    Y_tmp = Y_train[resample_idx]

    fit = pclogit(as.matrix(X_tmp), as.matrix(Y_tmp), alpha=0.8, group=group_size, lambda=lambdai)
    coeff = as.matrix(fit$beta)
    colnames(coeff) = lambdai
    write.csv(coeff, file = paste0(out_dir, '/pclogit/pclogit_coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

    fea_prob[, i] = I(coeff!=0)
  }
  fea_prob_sum = rowSums(fea_prob)/resample_time
  fea_prob1 = cbind(fea_prob, fea_prob_sum)
  colnames(fea_prob1)[resample_time+1]="ave"
  write.csv(fea_prob1, file = paste0(out_dir, '/pclogit/pclogit_prob_lam',ll*10,'.csv'), row.names = T)
}
