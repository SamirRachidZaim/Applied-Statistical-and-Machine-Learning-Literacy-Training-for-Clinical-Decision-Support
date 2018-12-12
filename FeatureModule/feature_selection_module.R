# set working directory
setwd("~/Dropbox/08. Fall 2018/SIE577/Final Project")
# setwd("/extra/akim127/SIE577")

library(caret)
library(MASS)
library(glmnet)
library(e1071)
library(randomForest)
library(rpart)
library(nnet)
library(NeuralNetTools)

################################################################
# Bladder Cancer Data
################################################################
# there are some columns that are perfectly correlated
# the design matrix is rank deficient

# read the data
b <- read.csv("bladder.csv", header=T)
dim(b)
# [1] 4016   95

b <- b[,-1]

# remove columns with all zeros
b <- b[, colSums(b != 0) > 0]
dim(b)
# [1] 4016   74

###########################################################################
# Feature selection simulations
###########################################################################
# create the training set/test set
# randomly select 10 patients from the R group and 10 patients from the NR group
set.seed(200)
rsample <- sample(1:20, 10)
tr_r <- sort(rsample)
set.seed(201)
nrsample <- sample(21:40, 10)
tr_nr <- sort(nrsample)
sample <- c(tr_r, tr_nr)

train <- subset(b, b$id %in% sample)
test <- subset(b, !(b$id %in% sample))

dim(train)
# [1] 2011   74
dim(test)
# [1] 2005   74

x_train <- train[, 1:72]
y_train <- train[, 74]

x_train <- as.matrix(x_train)
y_train <- as.matrix(y_train)

# need to remove columns with all 0 values
colsum_x <- colSums(x_train)
colsum_x_ind <- c(which(colsum_x != 0))
keep <- names(colsum_x_ind)

# define x_train again
x_train <- x_train[, colSums(x_train != 0) > 0]
dim(x_train)
# [1] 2011   59

# find index of the columns that are kept
keep_ind <- which(names(test) %in% keep)

x_test <- test[, keep_ind]
y_test <- test[, 74]

x_test <- as.matrix(x_test)
y_test <- as.matrix(y_test)

###########################################################################
# 1. LDA
###########################################################################
# fit LDA using the training set
lda_fit1 <- lda(x_train, y_train)
# Warning message:
#   In lda.default(x, grouping, ...) : variables are collinear

lda_fit1_coef <- lda_fit1$scaling

# predictions
train_pred_lda1 <- predict(lda_fit1, x_train)$class
test_pred_lda1 <- predict(lda_fit1, x_test)$class

train_pred_mat11 <- cbind(train_pred_lda1, train[,73])
test_pred_mat11 <- cbind(test_pred_lda1, test[,73])

# calculate errors
train_err1 <- mean(y_train!=train_pred_lda1)
test_err1 <- mean(y_test!=test_pred_lda1)

train_err1
# [1] 0.1924416
test_err1
# [1] 0.4458853

# fit LDA using the test set
lda_fit2 <- lda(x_test, y_test)
# Warning message:
#   In lda.default(x, grouping, ...) : variables are collinear

lda_fit2_coef <- lda_fit2$scaling

# predictions
train_pred_lda2 <- predict(lda_fit2, x_test)$class
test_pred_lda2 <- predict(lda_fit2, x_train)$class

train_pred_mat12 <- cbind(train_pred_lda2, test[,73])
test_pred_mat12 <- cbind(test_pred_lda2, train[,73])

# calculate errors
train_err2 <- mean(y_test!=train_pred_lda2)
test_err2 <- mean(y_train!=test_pred_lda2)

train_err2
# [1] 0.2633416
test_err2
# [1] 0.4833416

###########################################################################
# 2. LASSO
###########################################################################
# cv using the training set
set.seed(202)
cv_lasso1 <- cv.glmnet(x_train, y_train, alpha=1, family="binomial")
lambda_lasso1 <- cv_lasso1$lambda.min
lambda_lasso1
# [1] 0.0004337507

# extract coefficients
lasso_fit1_coef <- coef(cv_lasso1, s=lambda_lasso1)

# fit LASSO using the training set
lasso_fit1 <- glmnet(x_train, y_train, alpha=1, family="binomial")

# predictions
train_pred_lasso1 <- predict(lasso_fit1, newx=x_train, s=lambda_lasso1, type="class")
test_pred_lasso1 <- predict(lasso_fit1, newx=x_test, s=lambda_lasso1, type="class")

train_pred_mat21 <- cbind(train_pred_lasso1, train[,73])
test_pred_mat21 <- cbind(test_pred_lasso1, test[,73])

# calculate errors
train_err1 <- mean(y_train!=train_pred_lasso1)
test_err1 <- mean(y_test!=test_pred_lasso1)

train_err1
# [1] 0.1899552
test_err1
# [1] 0.4379052

# cv using the test set
set.seed(203)
cv_lasso2 <- cv.glmnet(x_test, y_test, alpha=1, family="binomial")
lambda_lasso2 <- cv_lasso2$lambda.min
lambda_lasso2
# [1] 0.0003958419

# extract coefficients
lasso_fit2_coef <- coef(cv_lasso2, s=lambda_lasso2)

# fit LASSO using the test set
lasso_fit2 <- glmnet(x_test, y_test, alpha=1, family="binomial")

# predictions
train_pred_lasso2 <- predict(lasso_fit2, newx=x_test, s=lambda_lasso2, type="class")
test_pred_lasso2 <- predict(lasso_fit2, newx=x_train, s=lambda_lasso2, type="class")

train_pred_mat22 <- cbind(train_pred_lasso2, test[,73])
test_pred_mat22 <- cbind(test_pred_lasso2, train[,73])

# calculate errors
train_err2 <- mean(y_test!=train_pred_lasso2)
test_err2 <- mean(y_train!=test_pred_lasso2)

train_err2
# [1] 0.2568579
test_err2
# [1] 0.5002486

###########################################################################
# 3. Elastic net
###########################################################################
# cv using the training set
# set vectors of length same as the alpha seq to store the corresponding lambda value that minimizes CV
alpha_seq <- seq(0.9, 0.1, -0.1)
cvm_min <- lamb <- rep(0, 9)

for (k in 1:length(alpha_seq)){
  set.seed(204)
  enet_cv <- cv.glmnet(x_train, y_train, nfolds=10, family="binomial", alpha=alpha_seq[k])
  cvm_min[k] <- enet_cv$cvm[which.min(enet_cv$lambda)]
  lamb[k] <- enet_cv$lambda.min
}

# select alpha/lambda that minimizes mean cv
par_index1 <- which.min(cvm_min)
cvm_enet1 <- min(cvm_min)
alpha_enet1 <- alpha_seq[par_index1]
lambda_enet1 <- lamb[par_index1]

alpha_enet1
# [1] 0.9
lambda_enet1
# [1] 0.0003026756

# now, fit elastic net using the selected parameters alpha and lambda
enet_fit1 <- glmnet(x_train, y_train, family="binomial", standardize=FALSE, alpha=alpha_enet1)
enet_fit1_coef <- coef(enet_fit1, s=lambda_enet1)

# predictions
train_pred_enet1 <- predict(enet_fit1, newx=x_train, s=lambda_enet1, type="class")
test_pred_enet1 <- predict(enet_fit1, newx=x_test, s=lambda_enet1, type="class")

train_pred_mat31 <- cbind(train_pred_enet1, train[,73])
test_pred_mat31 <- cbind(test_pred_enet1, test[,73])

# calculate errors
train_err1 <- mean(y_train!=train_pred_enet1)
test_err1 <- mean(y_test!=test_pred_enet1)

train_err1
# [1] 0.1979115
test_err1
# [1] 0.4418953

# cv using the test set
# set vectors of length same as the alpha seq to store the corresponding lambda value that minimizes CV
alpha_seq <- seq(0.9, 0.1, -0.1)
cvm_min <- lamb <- rep(0, 9)

for (k in 1:length(alpha_seq)){
  set.seed(205)
  enet_cv <- cv.glmnet(x_test, y_test, nfolds=10, family="binomial", alpha=alpha_seq[k])
  cvm_min[k] <- enet_cv$cvm[which.min(enet_cv$lambda)]
  lamb[k] <- enet_cv$lambda.min
}

# select alpha/lambda that minimizes mean cv
par_index2 <- which.min(cvm_min)
cvm_enet2 <- min(cvm_min)
alpha_enet2 <- alpha_seq[par_index2]
lambda_enet2 <- lamb[par_index2]

alpha_enet2
# [1] 0.9
lambda_enet2
# [1] 0.0005814219

# fit elastic net using the test set
enet_fit2 <- glmnet(x_test, y_test, alpha=alpha_enet2, family="binomial")
enet_fit2_coef <- coef(enet_fit2, s=lambda_enet2)

# predictions
train_pred_enet2 <- predict(enet_fit2, newx=x_test, s=lambda_enet2, type="class")
test_pred_enet2 <- predict(enet_fit2, newx=x_train, s=lambda_enet2, type="class")

train_pred_mat32 <- cbind(train_pred_enet2, test[,73])
test_pred_mat32 <- cbind(test_pred_enet2, train[,73])

# calculate errors
train_err2 <- mean(y_test!=train_pred_enet2)
test_err2 <- mean(y_train!=test_pred_enet2)

train_err2
# [1] 0.2543641
test_err2
# [1] 0.5002486

################################################################
# 4. SVM
################################################################
# tuning using the training set
set.seed(206)
tune1 <- tune.svm(scale(x_train), as.factor(y_train), gamma = 10^(-3:3), cost = 10^(-3:1), 
                  kernel="polynomial", scale=FALSE)
par1 <- tune1$best.parameters
par1
#     gamma cost
# 17   0.1  0.1

# fit svm using the training set
svm_fit1 <- svm(scale(x_train), as.factor(y_train), gamma = par1[1], cost = par1[2], 
                method = "C-classification", kernel = "polynomial", scale=FALSE)

# predictions
train_pred_svm1 <- predict(svm_fit1, scale(x_train))
test_pred_svm1 <- predict(svm_fit1, scale(x_test))

train_pred_mat41 <- cbind(train_pred_svm1, train[,73])
test_pred_mat41 <- cbind(test_pred_svm1, test[,73])

# calculate errors
train_err1 <- mean(y_train!=train_pred_svm1)
test_err1 <- mean(y_test!=test_pred_svm1)

train_err1
# [1] 0.09497762
test_err1
# [1] 0.4563591

# tuning using the test set
set.seed(207)
tune2 <- tune.svm(scale(x_test), as.factor(y_test), gamma = 10^(-3:3), cost = 10^(-3:1), 
                  kernel="polynomial", scale=FALSE)
par2 <- tune2$best.parameters
par2
#     gamma cost
# 17   0.1  0.1

# fit svm using the test set
svm_fit2 <- svm(scale(x_test), as.factor(y_test), gamma = par2[1], cost = par2[2], 
                method = "C-classification", kernel = "polynomial", scale=FALSE)

# predictions
train_pred_svm2 <- predict(svm_fit2, scale(x_test))
test_pred_svm2 <- predict(svm_fit2, scale(x_train))

train_pred_mat42 <- cbind(train_pred_svm2, test[,73])
test_pred_mat42 <- cbind(test_pred_svm2, train[,73])

# calculate errors
train_err2 <- mean(y_test!=train_pred_svm2)
test_err2 <- mean(y_train!=test_pred_svm2)

train_err2
# [1] 0.125187
test_err2
# [1] 0.5221283

################################################################
# 5. Tree
################################################################
parameter <- rpart.control(xval=10, cp=0.01)

# fit tree using the training set
set.seed(208)
rp_fit1 <- rpart(group~., data=train[,-73], method="class", control=parameter)
printcp(rp_fit1)

train_pred_rp1 <- predict(rp_fit1, train[,-c(73:74)], type="class")
test_pred_rp1 <- predict(rp_fit1, test[,-c(73:74)], type="class")

train_pred_mat51 <- cbind(train_pred_rp1, train[,73])
test_pred_mat51 <- cbind(test_pred_rp1, test[,73])

train_err1 <- mean(y_train!=train_pred_rp1)
test_err1 <- mean(y_test!=test_pred_rp1)

train_err1
# [1] 0.1889607
test_err1
# [1] 0.484788

# fit tree using the test set
set.seed(209)
rp_fit2 <- rpart(group~., data=test[,-73], method="class", control=parameter)
printcp(rp_fit2)

train_pred_rp2 <- predict(rp_fit2, test[,-c(73:74)], type="class")
test_pred_rp2 <- predict(rp_fit2, train[,-c(73:74)], type="class")

train_pred_mat52 <- cbind(train_pred_rp2, test[,73])
test_pred_mat52 <- cbind(test_pred_rp2, train[,73])

train_err2 <- mean(y_test!=train_pred_rp1)
test_err2 <- mean(y_train!=test_pred_rp1)

train_err2
# [1] 0.125187
test_err2
# [1] 0.4853307

################################################################
# 6. Random Forest
################################################################
# fit random forest using the training set
set.seed(210)
rf_fit1 <- randomForest(x_train, as.factor(y_train), importance=T)
rf_fit1

# Call:
#   randomForest(x = x_train, y = as.factor(y_train), importance = T) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 7
# 
# OOB estimate of  error rate: 17.35%
# Confusion matrix:
#   NR   R class.error
# NR 832 176   0.1746032
# R  173 830   0.1724826

train_pred_rf1 <- predict(rf_fit1, x_train)
test_pred_rf1 <- predict(rf_fit1, x_test)

train_pred_mat61 <- cbind(train_pred_rf1, train[,73])
test_pred_mat61 <- cbind(test_pred_rf1, test[,73])

train_err1 <- mean(y_train!=train_pred_rf1)
test_err1 <- mean(y_test!=test_pred_rf1)

train_err1
# [1] 0
test_err1
# [1] 0.4279302

# fit random forest using the test set
set.seed(211)
rf_fit2 <- randomForest(x_test, as.factor(y_test), importance=T)
rf_fit2

# Call:
#   randomForest(x = x_test, y = as.factor(y_test), importance = T) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 7
# 
# OOB estimate of  error rate: 19.85%
# Confusion matrix:
#   NR   R class.error
# NR 835 167   0.1666667
# R  231 772   0.2303091

train_pred_rf2 <- predict(rf_fit2, x_test)
test_pred_rf2 <- predict(rf_fit2, x_train)

train_pred_mat62 <- cbind(train_pred_rf2, test[,73])
test_pred_mat62 <- cbind(test_pred_rf2, train[,73])

train_err2 <- mean(y_test!=train_pred_rf2)
test_err2 <- mean(y_train!=test_pred_rf2)

train_err2
# [1] 0
test_err2
# [1] 0.5191447

################################################################
# 6. Artificial Neural network
################################################################
train$nr <- ifelse(train$group == "NR", 1, 0)
train$r <- ifelse(train$group == "R", 1, 0)
test$nr <- ifelse(test$group == "NR", 1, 0)
test$r <- ifelse(test$group == "R", 1, 0)

trainxin<- as.data.frame(x_train)
trainxout <- data.frame(train$nr, train$r)

testxin <- as.data.frame(x_test)
testxout <- data.frame(test$nr, test$r)

set.seed(212)
ann_fit1 <- nnet(trainxin, trainxout, size=10, softmax=TRUE, 
                 maxit= 1000, abstol=1e-10, MaxNWts=3000)

train_pred_ann1 <- round(predict(ann_fit1, trainxin))
test_pred_ann1 <- round(predict(ann_fit1, testxin))

table11 <- abs(train_pred_ann1-trainxout)
table12 <- abs(train_pred_ann1-testxout)

train_err1 <- (sum(table11)/2)/nrow(trainxout)
test_err1 <- (sum(table12)/2)/nrow(testxout)

train_err1
# [1] 0.3137742
test_err1
# [1] 0.3154613

set.seed(213)
ann_fit2 <- nnet(testxin, testxout, size=10, softmax=TRUE, 
                 maxit= 1000, abstol=1e-10, MaxNWts=3000)

train_pred_ann2 <- round(predict(ann_fit2, testxin))
test_pred_ann2 <- round(predict(ann_fit2, trainxin))

table21 <- abs(train_pred_ann2-testxout)
table22 <- abs(train_pred_ann2-trainxout)

train_err2 <- (sum(table21)/2)/nrow(testxout)
test_err2 <- (sum(table22)/2)/nrow(trainxout)

train_err2
# [1] 0.2763092
test_err2
# [1] 0.276728

save.image("~/Dropbox/08. Fall 2018/SIE577/Final Project/feature_selection_module.RData")
