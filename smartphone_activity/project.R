##########################
#smart phone activity competition
#benchmarks
rm(list=ls(all=TRUE))
#load in data
setwd("/Users/Yuchong/Documents/Study/STAT 613/project")
xtrain = as.matrix(read.csv(file="training_data.csv",header=FALSE))
ytrain = as.matrix(read.csv(file="training_labels.csv",header=FALSE))
strain = as.matrix(read.csv(file="training_subjects.csv",header=FALSE))

xtest = as.matrix(read.csv(file="test_data.csv",header=FALSE))
stest = as.matrix(read.csv(file="test_subjects.csv",header=FALSE))
features=read.table(file="features.txt")

#Divide train data into train and validate
smp_size <- floor(0.6 * nrow(xtrain))
set.seed(1)
train_ind <- sample(seq_len(nrow(xtrain)), size = smp_size)
xtrain_t<-xtrain[train_ind,]
ytrain_t<-ytrain[train_ind,]
xtrain_v<-xtrain[-train_ind,]
ytrain_v<-ytrain[-train_ind,]

#naive Bayes classifier
require("e1071")
fitB = naiveBayes(x=xtrain_t,y=as.factor(ytrain_t))
predB = predict(fitB,newdata=xtrain_v,type="class")
mean(predB==as.factor(ytrain_v))

#5-nearest neighbor
require("class")
predK = knn(test=xtrain_v,train=xtrain_t,cl=as.factor(ytrain_t),k=5)
mean(predK==as.factor(ytrain_v))

#-multinomial logistic regression
library(nnet)
train_t<-as.data.frame(cbind(ytrain_t,xtrain_t))  
fitM<-multinom(ytrain_t~.,data=train_t,MaxNWts=84581)
predM<-predict(fitM,xtrain_t)
mean(predM==as.factor(ytrain_t))
predM<-predict(fitM,xtrain_v)
mean(predM==as.factor(ytrain_v))

#regularized multinomial logistic regression (rigid)
library(glmnet)
cvfit = cv.glmnet(xtrain_t,as.factor(ytrain_t),family = "multinomial",alpha=0)
cvfit$lambda.min
predict_rigidmlr<-predict(cvfit, xtrain_t, s = "lambda.min", type = "class")
mean(predict_rigidmlr==as.factor(ytrain_t))
predict_rigidmlr<-predict(cvfit, xtrain_v, s = "lambda.min", type = "class")
mean(predict_rigidmlr==as.factor(ytrain_v))
#regularized multinomial logistic regression (lasso)
library(glmnet)
cvfit = cv.glmnet(xtrain_t,as.factor(ytrain_t),family = "multinomial")
cvfit$lambda.min
predict_lassomlr<-predict(cvfit, xtrain_t, s = "lambda.min", type = "class")
mean(predict_lassomlr==as.factor(ytrain_t))
predict_lassomlr<-predict(cvfit, xtrain_v, s = "lambda.min", type = "class")
mean(predict_lassomlr==as.factor(ytrain_v))
#regularized multinomial logistic regression (grouped lasso)
library(glmnet)
cvfit = cv.glmnet(xtrain_t,as.factor(ytrain_t),family = "multinomial",type.multinomial = "grouped")
cvfit$lambda.min
predict_lassomlr<-predict(cvfit, xtrain_t, s = "lambda.min", type = "class")
mean(predict_lassomlr==as.factor(ytrain_t))
predict_lassomlr<-predict(cvfit, xtrain_v, s = "lambda.min", type = "class")
mean(predict_lassomlr==as.factor(ytrain_v))

#-linear SVMs

tune.out=tune(svm,train.x=xtrain_t,train.y=as.factor(ytrain_t),kernel ="linear",ranges =list(cost=c(0.001,0.01,0.1)))
bestmod_svm =tune.out$best.model
#cost=0.01 is the best model
predict_svm<-predict(bestmod_svm,xtrain_t)
mean(predict_svm==as.factor(ytrain_t))
predict_svm<-predict(bestmod_svm,xtrain_v)
mean(predict_svm==as.factor(ytrain_v))




#randomforest
library(randomForest)


ytrain_t_matrix<-as.matrix(ytrain_t)
colnames(ytrain_t_matrix)="response"
train_t<-cbind(ytrain_t_matrix,xtrain_t)
train_t<-data.frame(train_t)
train_t$response<-as.factor(train_t$response)

ytrain_v<-as.factor(ytrain_v)

mtry_array=c(12,24,48,96)
ntree_array=c(250,500,1000)

fit6_1 = randomForest(response~.,mtry=12,data=train_t,ntree=250,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_2 = randomForest(response~.,mtry=12,data=train_t,ntree=500,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_3 = randomForest(response~.,mtry=12,data=train_t,ntree=1000,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_4 = randomForest(response~.,mtry=24,data=train_t,ntree=250,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_5 = randomForest(response~.,mtry=24,data=train_t,ntree=500,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_6 = randomForest(response~.,mtry=24,data=train_t,ntree=1000,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_7 = randomForest(response~.,mtry=48,data=train_t,ntree=250,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_8 = randomForest(response~.,mtry=48,data=train_t,ntree=500,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_9 = randomForest(response~.,mtry=48,data=train_t,ntree=1000,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_10 = randomForest(response~.,mtry=96,data=train_t,ntree=250,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_11 = randomForest(response~.,mtry=96,data=train_t,ntree=500,xtest=data.frame(xtrain_v),ytest=ytrain_v)
fit6_12 = randomForest(response~.,mtry=96,data=train_t,ntree=1000,xtest=data.frame(xtrain_v),ytest=ytrain_v)
print(fit6_1)



fit6 = randomForest(response~.,mtry=sqrt(ncol(train)-1),data=train,ntree=1000)
print(fit6)
fit6$predicted
fit6_2 = randomForest(response~.,mtry=sqrt(ncol(train)-1),data=train,ntree=2000)
print(fit6_2)
fit6_2$predicted
fit6_3 = randomForest(response~.,mtry=sqrt(ncol(train)-1),data=train,ntree=500)
print(fit6_2)
fit6_2$predicted

set.seed(1023)
fit6_4 = randomForest(response~.,mtry=sqrt(ncol(train)-1),data=train,ntree=1000)
# random forest
predict_rf<-predict(fit6_4,newdata=data.frame(xtest))
predict_rf= as.numeric(predict_rf)


cols = c("Id","Prediction")
submitS = cbind(1:length(predict_rf),predict_rf)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_rf2.csv",row.names=FALSE)

bestmtry <- tuneRF(xtrain_t, ytrain_t, stepFactor=1.5, improve=1e-5, ntree=500)
# Random Search
control <- trainControl(method="repeatedcv", number=5, repeats=1, search="random")
mtry <- sqrt(ncol(train_t)-1)
metric <- "Accuracy"
rf_random <- train(response~., data=train_t, method="rf", metric=metric, tuneLength=2, trControl=control)
print(rf_random)
plot(rf_random)

# bagging
fit_bagging = randomForest(response~.,mtry=ncol(train)-1,data=train,ntree=1000)
print(fit_bagging)
fit6$predicted
predict_bag<-predict(fit_bagging,newdata=data.frame(xtest))
predict_bag<- as.numeric(predict_bag)



cols = c("Id","Prediction")
submitS = cbind(1:length(predict_bag),predict_bag)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_bag.csv",row.names=FALSE)

#adaboosting
library(adabag)
fit_ada<-boosting(response~.,data=train, boos = TRUE, mfinal = 100, coeflearn ="Breiman")
predict_ada<-predict(fit_ada,newdata=data.frame(xtrain))
mean(as.numeric(predict_ada$class)==ytrain)
predict_ada<-predict(fit_ada,newdata=data.frame(xtest))
predict_ada<- as.numeric(predict_ada$class)

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_ada),predict_ada)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_ada.csv",row.names=FALSE)
#-xgboosting
library(xgboost)
dtrain <- xgb.DMatrix(data = xtrain_t,label = ytrain_t-1)
params <- list(booster = "gbtree", objective = "multi:softmax",num_class=6,eta=0.1, gamma=0, max_depth=4, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 400, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F,metrics="mlogloss")
xgbcv


#11.26
dtrain <- xgb.DMatrix(data = xtrain,label = ytrain-1)
dtrain <- xgb.DMatrix(data = xtrain_v,label = ytrain_v-1)
param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 6,
              max_depth = 6,
              eta = 0.05,
              gamma = 0.01, 
              subsample = 0.9,
              colsample_bytree = 0.8, 
              min_child_weight = 4,
              max_delta_step = 1
)
param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 6,
              max_depth = 6,
              eta = 0.01,
              gamma = 0., 
              subsample = 1,
              colsample_bytree = 1, 
              min_child_weight = 1,
              max_delta_step = 1
)
cv.nround = 1000
cv.nfold = 5
mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
               nfold=cv.nfold, nrounds=cv.nround,
               verbose = T,early_stop_round=10, maximize=FALSE)
mdcv$evaluation_log[,test_mlogloss_mean]
min(mdcv$evaluation_log[,test_mlogloss_mean])
which.min(mdcv$evaluation_log[,test_mlogloss_mean])
md2 <- xgb.train(data=dtrain, params=param, nrounds=100, nthread=6)

predict_xgb = predict(md2,newdata=xtrain_t)
pred <- matrix(predict_xgb, ncol=6, byrow=TRUE)
pred_labels <- max.col(pred)
mean(pred_labels==ytrain_t)

plot(mdcv$evaluation_log[,iter],mdcv$evaluation_log[,test_mlogloss_mean])

predict_xgb = predict(md2,newdata=xtest)
pred <- matrix(predict_xgb, ncol=6, byrow=TRUE)
pred_labels <- max.col(pred)
mean(pred_labels==ytrain)






xgfit= xgboost(data=data.matrix(xtrain_t),label=ytrain_t-1,max.depth=6,eta=.3,objective="multi:softmax",num_class=6,nround=87,verbose=0)
predict_xgb = round(predict(xgfit,newdata=xtrain_t))
mean(predict_xgb==(ytrain_t-1))
predict_xgb = round(predict(xgfit,newdata=xtrain_v))
mean(predict_xgb==(ytrain_v-1))

#-xgboosting with cross-validation
library(xgboost)
library(caret)
ControlParamteres <- trainControl(method = "cv",
                                  number = 5,
                                  savePredictions = TRUE,
                                  classProbs = TRUE
)
#(1)
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=0.7,
                               max_depth=c(4,6,8),
                               nrounds=200,
                               gamma=0,
                               min_child_weight=1,
                               subsample=1
)
#(2)
parametersGrid <-  expand.grid(eta = 0.3, 
                               colsample_bytree=1,
                               max_depth=6,
                               nrounds=100,
                               gamma=0,
                               min_child_weight=1,
                               subsample=1
)
#(3)
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=1,
                               max_depth=c(4,6,8),
                               nrounds=200,
                               gamma=c(1,5),
                               min_child_weight=1,
                               subsample=0.5
)

#(4)
parametersGrid <-  expand.grid(eta = 0.01, 
                               colsample_bytree=1,
                               max_depth=6,
                               nrounds=300,
                               gamma=0,
                               min_child_weight=1,
                               subsample=1
)

#(5)
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=1,
                               max_depth=6,
                               nrounds=200,
                               gamma=c(1,5),
                               min_child_weight=1,
                               subsample=1
)

set.seed(1023)
ControlParamteres <- trainControl(method = "cv",
                                  number = 5,
                                  savePredictions = TRUE,
                                  classProbs = TRUE
)
#11.27
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=1,
                               max_depth=c(4,6,8),
                               nrounds=200,
                               gamma=c(1,5),
                               min_child_weight=1,
                               subsample=0.8
)

modelxgboost <- caret::train(response~.,
                             data = train,
                             trControl = ControlParamteres,
                             tuneGrid = parametersGrid,
                             method = "xgbTree")
xgfit= xgboost(data=data.matrix(xtrain),label=ytrain-1,max.depth=6,eta=.1,objective="multi:softmax",num_class=6,nround=200)
colnames(xtrain)<-features[,2]
xgb.imp<-xgb.importance(colnames(xtrain), model = xgfit)
xgb.plot.importance(xgb.imp,10)
set.seed(1023)
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=1,
                               max_depth=c(3,4,5,6,7,8),
                               nrounds=200,
                               gamma=0,
                               min_child_weight=1,
                               subsample=1
)

modelxgboost2 <- caret::train(response~.,
                             data = train,
                             trControl = ControlParamteres,
                             tuneGrid = parametersGrid,
                             method = "xgbTree")



xgb.importance(colnames(train[,-1]), model = modelxgboost)
save(modelxgboost, file = "my_model1.rda")

ytrain_t_matrix<-as.matrix(ytrain_t)
colnames(ytrain_t_matrix)="response"
train_t<-cbind(ytrain_t_matrix,xtrain_t)
train_t<-data.frame(train_t)
train_t$response<-as.factor(train_t$response)
levels(train_t$response) <- make.names(levels(factor(train_t$response)))

ytrain_v<-as.factor(ytrain_v)

ytrain_matrix<-as.matrix(ytrain)
colnames(ytrain_matrix)="response"
train<-cbind(ytrain_matrix,xtrain)
train<-data.frame(train)
train$response<-as.factor(train$response)
levels(train$response) <- make.names(levels(factor(train$response)))
#ytrain_v_matrix<-as.matrix(ytrain_v)
#colnames(ytrain_v_matrix)="response"
#train_v<-cbind(ytrain_v_matrix,xtrain_v)


modelxgboost <- caret::train(response~.,
                      data = train,
                      trControl = ControlParamteres,
                      tuneGrid = parametersGrid,
                      method = "xgbTree")

predictions<-predict(modelxgboost,xtrain)
table(predictions=predictions,actual=ytrain)
str(predictions)

predict_xgb_cv3= predict(modelxgboost,xtest)
predict_xgb_cv3 = as.numeric(predict_xgb_cv3)

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_xgb_cv3),predict_xgb_cv3)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_xgboosting_cv3.csv",row.names=FALSE)
#Test the future
#5-nearest neighbor
require("class")
predK = knn(test=xtest,train=xtrain,cl=as.factor(ytrain),k=5)

cols = c("Id","Prediction")
submitK = cbind(1:length(predK),predK)
colnames(submitK) = cols
write.csv(submitK,file="benchmark_KNN.csv",row.names=FALSE)


#naive Bayes classifier
require("e1071")
fitB = naiveBayes(x=xtrain,y=as.factor(ytrain))
predB = predict(fitB,newdata=xtest,type="class")

cols = c("Id","Prediction")
submitB = cbind(1:length(predB),predB)
colnames(submitB) = cols
write.csv(submitB,file="benchmark_NB.csv",row.names=FALSE)

#multinomial logistic regression
library(nnet)
colnames(ytrain) <- c("ytrain")
train<-as.data.frame(cbind(ytrain,xtrain)) 

fitM<-multinom(ytrain~.,data=train,MaxNWts=84581)
predM<-predict(fitM,xtest)

cols = c("Id","Prediction")
submitM = cbind(1:length(predM),predM)
colnames(submitM) = cols
write.csv(submitM,file="benchmark_MLR.csv",row.names=FALSE)

#regularized multinomial logistic regression (grouped lasso)
library(glmnet)
cvfit = cv.glmnet(xtrain,as.factor(ytrain),family = "multinomial",type.multinomial = "grouped")
cvfit$lambda.min
predict_lassogmlr<-predict(cvfit, xtest, s = "lambda.min", type = "class")

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_lassogmlr),predict_lassogmlr)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_lassogmlr.csv",row.names=FALSE)

#regularized multinomial logistic regression (lasso)
library(glmnet)
cvfit = cv.glmnet(xtrain,as.factor(ytrain),family = "multinomial")
cvfit$lambda.min
predict_lassomlr<-predict(cvfit, xtest, s = "lambda.min", type = "class")

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_lassomlr),predict_lassomlr)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_lassomlr.csv",row.names=FALSE)

#regularized multinomial logistic regression (ridge)
library(glmnet)
cvfit = cv.glmnet(xtrain,as.factor(ytrain),family = "multinomial",alpha=0)
cvfit$lambda.min
predict_ridgemlr<-predict(cvfit, xtest, s = "lambda.min", type = "class")

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_ridgemlr),predict_ridgemlr)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_ridgemlr.csv",row.names=FALSE)

#linear svms
tune.out=tune(svm,train.x=xtrain,train.y=as.factor(ytrain),kernel ="linear",ranges =list(cost=c(0.001,0.01,0.1)))
bestmod_svm =tune.out$best.model
predict_svm<-predict(bestmod_svm,xtest)

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_svm),predict_svm)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_linearSVM.csv",row.names=FALSE)

#linear svms_2
tune.out=tune(svm,train.x=xtrain,train.y=as.factor(ytrain),kernel ="linear",ranges =list(cost=c(0.005,0.01,0.02,0.04)))
bestmod_svm =tune.out$best.model
predict_svm<-predict(bestmod_svm,xtest)

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_svm),predict_svm)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_linearSVM_2.csv",row.names=FALSE)

#xgboosting
xgfit= xgboost(data=xtrain,label=ytrain-1,max.depth=6,eta=.3,objective="multi:softmax",num_class=6,nround=100,verbose=0)
predict_xgb = round(predict(xgfit,newdata=xtest))+1

cols = c("Id","Prediction")
submitS = cbind(1:length(predict_xgb),predict_xgb)
colnames(submitS) = cols
write.csv(submitS,file="benchmark_xgboosting2.csv",row.names=FALSE)


#xgboosting_cross validation
library(mlr)
ytrain_t_matrix<-as.matrix(ytrain_t)
colnames(ytrain_t_matrix)="response"
train_t<-cbind(ytrain_t_matrix,xtrain_t)
ytrain_v_matrix<-as.matrix(ytrain_v)
colnames(ytrain_v_matrix)="response"
train_v<-cbind(ytrain_v_matrix,xtrain_v)
train_t<-data.frame(train_t)
train_t[,1]<-as.factor(train_t[,1])
traintask <- makeClassifTask (data = train_t,target = "response")
train_v<-data.frame(train_v)
train_v[,1]<-as.factor(train_v[,1])
testtask <- makeClassifTask (data = train_v,target = "response")


traintask <- createDummyFeatures (obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective = "multi:softprob",num_class=6, eval_metric="mlogloss", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)
#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = mmce, par.set = params, control = ctrl, show.info = T)
mytune$y 
#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)
confusionMatrix(xgpred$data$response,xgpred$data$truth)
