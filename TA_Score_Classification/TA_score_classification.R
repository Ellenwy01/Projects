
library(ISLR2)
tae<-read.csv("tae.data", header=FALSE)
ta<-tae

attach(ta)

#Explore data
dim(ta)
summary(ta)
str(ta)

#Rename the column name-------------------------------------------------------------------------------------------------------------------------------
names(ta)<-c("Native_English_speaker","Course_instructor",
             "Course","Semester","Class_size","Class_attribute")
head(ta)

#Prepare ta_num for unfactorized variables 
ta_num<-ta
str(ta_num)

#Check the cor----------------------------------------------------------------------------------------------------------------------------------------
cordata<-round(cor(ta_num),3)
cordata
write.csv(cordata, "Correlation_between_Variables.csv")



#Factor the class-------------------------------------------------------------------------------------------------------------------------------------
str(ta)
var<-c("Native_English_speaker","Course_instructor","Course","Semester",
       "Class_attribute")
ta[,var]<-lapply(ta[,var], factor)
str(ta)
summary(ta)

#Check class balance
summary(ta$Class_attribute)

#Split into trainig data and test set---------------------------------------------------------------------------------------------------------------- 
set.seed(123)
train<-sample(1:nrow(ta), nrow(ta)*0.8)

ta.train<-ta[train, ]
ta.test<-ta[-train, ]
#Check the dim of both training and test set
dim(ta.train)
dim(ta.test)

xtrain<-subset(ta.train, select = -Class_attribute)
xtest<-subset(ta.test, select = -Class_attribute)
ytrain<-ta.train$Class_attribute
ytest<-ta.test$Class_attribute

summary(ta.train$Class_attribute)
# numerical data for multinomial logistics regression, lda, and qda
set.seed(123)
ta_num_cor<-ta_num
ta_num_cor_train<-sample(1:nrow(ta_num), nrow(ta_num)*0.8)
ta_num_cor.train<-ta_num[ta_num_cor_train, ]

cordata_train<-round(cor(ta_num_cor.train),3)
cordata_train
write.csv(cordata_train, "Correlation_between_Variables_train.csv")

var2<-c("Class_attribute")
ta_num$Class_attribute<-factor(ta_num$Class_attribute)
str(ta_num)
set.seed(123)
train.num<-sample(1:nrow(ta_num), nrow(ta_num)*0.8)

ta_num.train<-ta_num[train.num, ]
ta_num.test<-ta_num[-train.num, ]

xtrain_num<-subset(ta_num.train, select = -Class_attribute)
xtest_num<-subset(ta_num.test, select = -Class_attribute)
ytrain_num<-ta_num.train$Class_attribute
ytest_num<-ta_num.test$Class_attribute



#-----------------------------------------------------------------------------------------------------------------------------------------------------
# 1.KNN
#generally try k=square root of obs, k=floor(sqrt(151)) ~ 12.3~12
library(class)
library(e1071)
library(caret)
#Try knn model with k=12
set.seed(123)
knn12<-knn(train=ta.train, test=ta.test, cl=ytrain, k=12)
confusionMatrix(ytest, knn12)
#Accuarcy: 35.48%

#k=13
set.seed(123)
knn13<-knn(train=ta.train, test=ta.test, cl=ytrain, k=13)
confusionMatrix(ytest, knn13)
#Accuarcy:41.94%

#Tune with 5-fold_CV
set.seed(123)
knn.cross<-tune.knn(xtrain,ytrain,k=1:20,
                   tunecontrol=tune.control(cross=5))
#Tune with Bootstraping
set.seed(123)
knn.boot<-tune.knn(xtrain,ytrain,k=1:20,
                   tunecontrol=tune.control(sampling="boot"))

#summary
set.seed(123)
summary(knn.cross)
summary(knn.boot)
plot(knn.boot)
plot(knn.tune)
#Both result with Best k=1
set.seed(123)
pred_knn<-knn(train=ta.train, test=ta.test, cl=ytrain, k=1)

confusionMatrix(ytest, pred_knn)
capture.output(confusionMatrix(ytest, pred_knn), file="Confusion_matrix_knn.txt")
#Accuracy: 70.97%


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# 2.SVM
set.seed(123)
library(e1071)
library(caret)
##Poly 
#Tune cost, degree, and gamma
set.seed(123)
tune.svm_poly<-tune(svm,Class_attribute~., data=ta.train,kernel="poly",
                ranges=list(degree=c(2,3,4,5),
                            cost=seq(0.01,10, length=100),
                            gramma=10^(-5:2)),
                tunecontrol=tune.control(cross=5))
tune.svm_poly$best.parameters
summary(tune.svm_poly)
tune.svm_poly$best.model

#training error/acc:
pred_svm_poly_train<-predict(tune.svm_poly$best.model, newdata=ta.train)
confusionMatrix(pred_svm_poly_train, ytrain)$overall[[1]]
#35.83%

capture.output(tune.svm_poly$best.model,file="svm_poly_bestmodel.txt")
pred_svm_poly<-predict(tune.svm_poly$best.model, newdata=ta.test)
confusionMatrix(pred_svm_poly,ytest)
capture.output(confusionMatrix(pred_svm_poly,ytest), file="Confusion_matrix_svm_poly.txt")
#Test set Accuracy: 29.03%


#Linear: Tune cost
set.seed(123)
tune.svm_lin<-tune(svm, Class_attribute~. , data=ta.train, kernel="linear",
                   ranges=list(cost=seq(0.01,10, length=100)),
                   tunecontrol=tune.control(cross=5))

summary(tune.svm_lin)
tune.svm_lin$best.parameters
pred_svm_lin_train<-predict(tune.svm_lin$best.model, newdata=ta.train)
confusionMatrix(pred_svm_lin_train, ytrain)$overall[[1]]
#training acc: 83.3%
capture.output(tune.svm_lin$best.model, file="svm_linear_bestmodel")

plot(tune.svm_lin)
pred_svm_lin<-predict(tune.svm_lin$best.model, newdata=ta.test)
confusionMatrix(pred_svm_lin,ytest)
capture.output(confusionMatrix(pred_svm_lin,ytest), file="Confusion_matrix_svm_lin.txt")
#Accuracy:58.06%

#Radial: Tune cost, gamma
set.seed(123)
tune.svm_rad<-tune(svm, Class_attribute~. , data=ta.train,kernel="radial",
                   ranges=list(cost=seq(0.01,10, length=100),
                               gamma=10^(-5:2)),
                   tunecontrol=tune.control(cross=5))

summary(tune.svm_rad)
capture.output(summary(tune.svm_rad), file="Summary_SVM_Radial.txt")
plot(tune.svm_rad)
pred_svm_rad_train<-predict(tune.svm_rad$best.model, newdata=ta.train)
confusionMatrix(pred_svm_rad_train, ytrain)$overall[[1]]
#training acc:89.17%
pred_svm_rad<-predict(tune.svm_rad$best.model, newdata=ta.test)
confusionMatrix(pred_svm_rad,ytest)
tune.svm_rad$best.parameters
capture.output(confusionMatrix(pred_svm_rad,ytest), file="Confusion_matrix_svm_rad.txt")
capture.output(tune.svm_rad$best.model,file="svm_rad_bestmodel.txt")


#Accuracy:61.29%

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Random Forest
#-------------------1. Only tune mtry------------------------------------------------------------------------------
library(MASS)
library(randomForest)
library(e1071)
library(caret)
set.seed(123)
for (i in 1:5){
  set.seed(123)
  rf_mtry<-randomForest(Class_attribute~., data=ta.train, mtry=i, importance=TRUE, ntree=100)
  pred.rf<-predict(rf_mtry, ta.test)
  print(confusionMatrix(pred.rf, ytest))
}

#mtry=1
set.seed(123)
best.mtry<- rf_mtry<-randomForest(Class_attribute~., data=ta.train, mtry=1, importance=TRUE, ntree=100)
pred.best.mtry<-predict(best.mtry, ta.test)
confusionMatrix(pred.best.mtry, ytest)

varImpPlot(best.mtry)
best.mtry$importance

#accuray:61.29%

#------------------- 2. With 5-fold CV, Tune:mtry,maxnodes,ntree ------------------------------------------------
library(e1071)
library(caret)
library(randomForest)
set.seed(123)

#Tune mtry
trControl<-trainControl(method = "cv", number = 5, search ="grid")
tuneGrid <- expand.grid(.mtry = c(1: 5))
rf_mtry<-train(Class_attribute~.,
               data = ta.train,
               method = "rf",
               metric = "Accuracy",
               tuneGrid = tuneGrid,
               trControl = trControl,
               importance = TRUE
               )
plot(rf_mtry)
print(rf_mtry)

best.mtry<-rf_mtry$bestTune$mtry
best.mtry
tuneGrid=expand.grid(.mtry = best.mtry)

#Tune maxnode
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best.mtry)
for (maxnodes in c(5: 25)) {
  set.seed(123)
  rf_maxnode <- train(Class_attribute~.,
                      data = ta.train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      maxnodes = maxnodes)
                      
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#Tune ntree

store_maxtrees <- list()
for (ntree in c(50,100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(123)
  rf_maxtrees <- train(Class_attribute~.,
                       data = ta.train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       maxnodes=16,
                       ntree = ntree)
                       
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

#Fit RF model with result: mtry=5,maxnodes=16,ntree=400
set.seed(123)
trControl<-trainControl(method = "cv", number = 5)
fit.rf<- train(Class_attribute~.,
               data = ta.train,
               method = "rf",
               metric = "Accuracy",
               tuneGrid = tuneGrid,
               trControl = trControl,
               importance = TRUE,
               maxnodes=16,
               ntree = 400)


#Plot result of fit model
jpeg(file="rf_tune_mtry,ntree,maxnodes_cv.jpeg", width=400, height=300, quality=100)
plot(fit.rf$finalModel)
dev.off()

varImpPlot(fit.rf$finalModel)
importance(fit.rf$finalModel)
jpeg(file="varImpPlot_rf_3.jpeg",width=1000, height=700, quality=100)
varImpPlot(fit.rf$finalModel)
dev.off()
capture.output(importance(fit.rf$finalModel), file="importance_rf_3.txt")

rf.pred_train<-predict(fit.rf, ta.train)
confusionMatrix(rf.pred_train, ytrain)$overal[[1]]
#training acc: 78.3%

#Predict and result
rf.pred<-predict(fit.rf, ta.test)
confusionMatrix(rf.pred, ytest)
capture.output(confusionMatrix(rf.pred, ytest), file="Confusion_matrix_rf_tune3.txt")


#Accuracy: 64.52%

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#4. Logistics Regression 

library(nnet)
set.seed(123)
# Training the multinomial model
multinom.fit <- multinom(Class_attribute~., data = ta_num.train)

#Checking the model
summary(multinom.fit)
exp(coef(multinom.fit))
#Save model summary
sink("summary_mutinom.txt")
summary(multinom.fit)
sink()

#Predict
library(caret)
set.seed(123)
#training data
pred_multinom1<-predict(multinom.fit, newdata=ta_num.train)
confusionMatrix(pred_multinom1, ytrain_num)
#Training accuracy: 53.33%

#test data
pred_multinom2<-predict(multinom.fit, newdata=ta_num.test)
result.multinom<-confusionMatrix(pred_multinom2, ytest_num)
result.multinom
capture.output(result.multinom, file="Confusion_matrix_Multinomial_logistics_regression.txt")
#Accuracy: 54.48%

#--------With 5-fold CV---------------------------------------------------------------------------
trControl<-trainControl(method = "cv", number = 5, search ="grid")
set.seed(123)  
multinom.fit_cv <- train(Class_attribute ~ ., data = ta_num.train, method = "multinom", metric="Accuracy",
             trControl = trControl)

pred_multinom_cv<-predict(multinom.fit_cv, newdata=ta_num.test)
result.multinom_cv<-confusionMatrix(pred_multinom_cv, ytest_num)
result.multinom_cv
capture.output(result.multinom_cv, file="Confusion_matrix_Multinomial_logistics_regression_cv.txt")
#Accuracy: 54.84%

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#5.LDA
set.seed(123)
trControl<-trainControl(method = "cv", number = 5, summaryFunction = multiClassSummary())
lda.fit<-train(Class_attribute ~ ., data = ta_num.train, method = "lda", metric="Accuracy",
               trControl = trControl)
pred_lda_train<-predict(lda.fit, ta_num.train)
confusionMatrix(pred_lda_train, ytrain_num)$overall[[1]]
#training acc:56.67%
pred_lda<-predict(lda.fit, newdata=ta_num.test)
confusionMatrix(pred_lda, ytest_num)
capture.output(confusionMatrix(pred_lda, ytest_num), file="Confusion_matrix_lda_cv.txt")
lda.fit$finalModel
#Accuracy: 58.06%

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#6.QDA
set.seed(123)
trControl<-trainControl(method = "cv", number = 5,summaryFunction = multiClassSummary)
qda.fit<-train(Class_attribute ~ ., data = ta_num.train, method = "qda",metric="Accuracy", 
               trControl = trControl)

pred_qda_train<-predict(qda.fit, ta_num.train)
confusionMatrix(pred_qda_train, ytrain_num)$overall[[1]]
#training acc: 57.5%

pred_qda<-predict(qda.fit, newdata=ta_num.test)
confusionMatrix(pred_qda, ytest_num)
capture.output(confusionMatrix(pred_qda, ytest_num), file="Confusion_matrix_qda_cv.txt")

#Accuracy: 54.84%

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#7.Classification Tree
library(tree)
set.seed(123)
tree<- tree(Class_attribute~., data = ta.train)
cv.tree<-cv.tree(tree, FUN = prune.misclass)
plot(cv.tree$size, cv.tree$dev, type="b")
plot(cv.tree$k, cv.tree$dev, type="b")
tree.min<-cv.tree$size[which.min(cv.tree$dev)]
tree.min

#without prune
tree.pred<-predict(tree, newdata=ta.test, type="class")
confusionMatrix(tree.pred, ytest)
plot(tree)
text(tree, pretty=0)
#With prune
prune_tree<-prune.misclass(tree, best=tree.min)
jpeg(file="prune_tree.jpeg", width=1000, height=800, quality=100)
plot(prune_tree)
text(prune_tree, pretty=0)
dev.off()
summary(prune_tree)
summary(prune_tree)$used

prune_tree.pred_train<-predict(prune_tree, newdata=ta.train, type="class")
confusionMatrix(prune_tree.pred_train, ytrain)$overall[[1]]
#training acc: 75%
prune_tree.pred<-predict(prune_tree, newdata=ta.test, type="class")
confusionMatrix(prune_tree.pred, ytest)
capture.output(confusionMatrix(prune_tree.pred, ytest), file="Confusion_matrix_tree_cv.txt")

#Same Accuracy: 51.61%

#---------------using rpart instead of tree function-------------------------------------------------------------------------

set.seed(123)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(caret)
#Tuning-----------
set.seed(123)
trcontrol=trainControl(method= "cv",number=5)
#only tune cp
set.seed(123)
tune_rpart1<-train(Class_attribute~.,data=ta.train,
                   method="rpart",trControl=trcontrol,tuneLength=10)
tune_rpart1
bestcp<-tune_rpart1$bestTune$cp
bestcp

fancyRpartPlot(tune_rpart1$finalModel)
pred_tune_rpart1_train<-predict(tune_rpart1, ta.train, type="raw")
confusionMatrix(pred_tune_rpart1_train, ytrain)$overall[[1]]
#training:55.83%
pred_tune_rpart1<-predict(tune_rpart1,newdata=ta.test,type="raw")
confusionMatrix(pred_tune_rpart1, ytest)
#acc: 48.39%

#only tune max tree depth
set.seed(123)
tune_rpart2<-train(Class_attribute~.,data=ta.train,metric = "Accuracy",
                   method="rpart2",trControl=trcontrol,tuneLength=10)
tune_rpart2
bestmaxdepth<-tune_rpart2$bestTune$maxdepth
jpeg(file="Classification_tree_rpart_maxdepth.jpeg",width=1000, height=700, quality=100)
fancyRpartPlot(tune_rpart2$finalModel,main="Classification Tree")
dev.off()
rpart.plot(tune_rpart2$finalModel,box.palette="RdBu", shadow.col="gray", nn=TRUE)

pred_tune_rpart2_train<-predict(tune_rpart2, ta.train, type="raw")
confusionMatrix(pred_tune_rpart2_train, ytrain)$overall[[1]]
#trainig:60%

pred_tune_rpart2<-predict(tune_rpart2,newdata=ta.test,type="raw")
confusionMatrix(pred_tune_rpart2, ytest)
capture.output(confusionMatrix(pred_tune_rpart2, ytest), file="Confusion_matrix_tree_rpart_maxdepth.txt")
#Accuracy=54.84%

#untune
tree_rpart <- rpart(Class_attribute~., data = ta.train, method = 'class')
jpeg(file="Classification_tree_fit_rpart.jpeg",width=1000, height=700, quality=100)
rpart.plot(tree_rpart)
dev.off()
rpart.plot(tree_rpart)
plotcp(tree_rpart)
summary(tree_rpart)
#Determine the value of the complexity parameter that produces
#the lowest CV error

cp.min <- tree_rpart$cptable[which.min(tree_rpart$cptable[,"xerror"]),"CP"]
prune_tree_rpart <- prune(tree_rpart, cp = cp.min)
pred.tree_rpart<-predict(prune_tree_rpart, newdata=ta.test, type="class")
confusionMatrix(pred.tree_rpart, ytest)
capture.output(confusionMatrix(pred.tree_rpart, ytest), file="Confusion_matrix_tree_rpart.txt")
#Accuracy:38.71%
#-----------------
