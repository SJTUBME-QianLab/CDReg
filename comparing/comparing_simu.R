Args <- commandArgs()
data_name = Args[3]
seed = Args[4]
model = Args[5]
# fold = Args[5]
# data_name = 'covGau_de1.0_seed2023'
# seed = 2023
# fold = 0
library("glmnet")
library("grpreg")
library("SGL")
library("pclogit")

data_dir = paste0("./../data/simulation/", data_name, "/")
out_dir = paste0("./../results/", data_name, "/", model, "/s", seed, "/")
if (!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}
method = "gaussian"
# method = 'binomial'
set.seed(seed)

#### load data ####
X = read.table(paste0(data_dir, "X_normL2.csv"), sep=",", header=F)
Y = read.table(paste0(data_dir, "Y.csv"), sep=",", header=F)
# test_idx0 = read.table(paste0(data_dir, "/test_idx/", fold, ".txt"), header=F)
# test_idx = as.matrix(test_idx0 + 1)
test_idx = as.matrix(0)
train_idx = setdiff(1: dim(X)[1], test_idx)
X_train = X[train_idx, ]
Y_train = Y[train_idx, ]

basic_info = read.table(paste0(data_dir, "basic_info.csv"), sep=",", header=T)
# group = read.table(paste0(data_dir, "gp_label.csv"), sep=",", header=F)
group = as.numeric(as.matrix(basic_info$gp_label))
fea_name = basic_info$fea_name

#### Resample 100 times for Selection Probability ####
p = dim(X_train)[2]
resample_time = 1
resample_rate = 1
fea_prob = matrix(0, p, resample_time)
fea_rank = matrix(0, p, resample_time)

if(model == 'Enet0.8'){
  #### Enet 0.8 ####
  cvfit = cv.glmnet(as.matrix(X_train), as.matrix(Y_train), alpha=0.8, family=method)
  # lambda_list = cvfit$lambda[c(10,20,30,40,50,60,70,80,90,100)]
  # lambda_list = c(cvfit$lambda.1se)
  lambda_list = c(cvfit$lambda[length(cvfit$lambda)])  # last one (smallest)
  # lambda_list = c(0.0005)
  write.csv(cvfit$lambda, file = paste0(out_dir, '/lambda_list.csv'), row.names = T)
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
      write.csv(coeff, file = paste0(out_dir, '/coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

      fea_prob[, i] = I(coeff[-1, ]!=0)
    }
    fea_prob_sum = rowSums(fea_prob)/resample_time
    fea_prob1 = cbind(fea_prob, fea_prob_sum)
    colnames(fea_prob1)[resample_time+1]="ave"
    write.csv(fea_prob1, file = paste0(out_dir, '/prob_lam',ll*10,'.csv'), row.names = T)
  }

} else if (model == 'LASSO'){
  #### LASSO ####
  cvfit = cv.glmnet(as.matrix(X_train), as.matrix(Y_train), alpha=1, family=method)
  # lambda_list = cvfit$lambda[c(10,20,30,40,50,60,70,80,90,100)]
  # lambda_list = c(cvfit$lambda.1se)
  lambda_list = c(cvfit$lambda[length(cvfit$lambda)])  # last one (smallest)
  # lambda_list = c(0.0005)
  write.csv(cvfit$lambda, file = paste0(out_dir, '/lambda_list.csv'), row.names = T)
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
      write.csv(coeff, file = paste0(out_dir, '/coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

      fea_prob[, i] = I(coeff[-1, ]!=0)
    }
    fea_prob_sum = rowSums(fea_prob)/resample_time
    fea_prob1 = cbind(fea_prob, fea_prob_sum)
    colnames(fea_prob1)[resample_time+1]="ave"
    write.csv(fea_prob1, file = paste0(out_dir, '/prob_lam',ll*10,'.csv'), row.names = T)
  }

} else if (model == 'SGLASSO'){
    #### SGL sparse group lasso ####
  if (method == "gaussian"){
    type = "linear"
  }else{
    return(message("wrong method"))
  }
  # cvfit = cvSGL(data=list(x=as.matrix(X_train), y=as.matrix(Y_train)),
  #               index=group, type=type)
  # lambda_list = cvfit$lambda[c(1:10)*2]
  # lambda_list = c(cvfit$lambda[length(cvfit$lambda)])  # total 20, last one (smallest)
  lambda_list = c(0.0001)
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
      write.csv(coeff, file = paste0(out_dir, '/coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

      fea_prob[, i] = I(coeff!=0)
    }
    fea_prob_sum = rowSums(fea_prob)/resample_time
    fea_prob1 = cbind(fea_prob, fea_prob_sum)
    colnames(fea_prob1)[resample_time+1]="ave"
    write.csv(fea_prob1, file = paste0(out_dir, '/prob_lam',ll*10,'.csv'), row.names = T)
  }

} else if (model == 'pclogit'){
  #### pclogit ####
  group_size = rep(0, max(group))
  for (k in 1:max(group)){
    group_size[k] = length(which(group==k))
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
      write.csv(coeff, file = paste0(out_dir, '/coef_lam',ll*10,'_re', i,'.csv'), row.names = T)

      fea_prob[, i] = I(coeff!=0)
    }
    fea_prob_sum = rowSums(fea_prob)/resample_time
    fea_prob1 = cbind(fea_prob, fea_prob_sum)
    colnames(fea_prob1)[resample_time+1]="ave"
    write.csv(fea_prob1, file = paste0(out_dir, '/prob_lam',ll*10,'.csv'), row.names = T)
  }

} else{
  print("wrong model")
}


