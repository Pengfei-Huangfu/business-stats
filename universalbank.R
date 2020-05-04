ubank<-read.csv("UniversalBank.csv")
head(ubank)
ubank.df<-ubank[,-c(1,5)]
head(ubank.df)
ubank.df$Education<-factor(ubank.df$Education, levels =c(1,2,3),
                           labels = c ("Undergrad", "Graduate", "Advanced/Professional"))
head(ubank.df)
library(neuralnet)
library(nnet)
library(ggplot2)
library(lattice)
library(caret)
dim(ubank.df)

train.index<-sample(c(1:dim(ubank.df)[1]),dim(ubank.df)[1]*0.6)
train.df<- ubank.df[train.index,]
valid.df<- ubank.df[-train.index,]

ubank.glm<- glm(PersonalLoan~.,binomial,train.df)
summary(ubank.glm)

head(train.df)
test<-ubank.df[,-c(2,3)]
head(train.df)

test<-ubank.df[,-c(1,2)]
head(train.df)
ubank.glm1<- glm(PersonalLoan~.,binomial,test)
summary(ubank.glm1)

ubankpred<-predict(ubank.glm,valid.df,type = "response")
data.frame(actual = valid.df$PersonalLoan[1:5], predicted = ubankpred[1:5])
predictions<- ifelse(ubankpred>0.5,1,0)
head(predictions)
cm<-table(valid.df$PersonalLoan,predictions>0.5)
confusionMatrix(as.factor(predictions),as.factor(valid.df$PersonalLoan))
cm
#neural net
head(train.df)
airline.pre<-preProcess(train.df[,c(1,2,3,4,5,6,7)],method = "range")
train.df[,c(1:7)]<- predict(airline.pre,train.df[,c(1:7)])
valid.df[,c(1:7)]<-predict(airline.pre,valid.df[,c(1:7)])
str(train.df)
train.df$Education<-as.numeric(train.df$Education)
nn<-neuralnet(as.factor(PersonalLoan)~.,data = train.df, linear.output = F , hidden = 3)
plot(nn, rep = "best")

valid.pre<-compute(nn,valid.df[,-ncol(valid.df)])
valid.out<-data.frame(actuals = valid.df[,ncol(valid.df)],
                      predicted = valid.pre$net.result)
valid.out$predicted.class<- ifelse(valid.out$predicted.2>0.5,1,0)
valid.out$actuals<- as.factor(valid.out$actuals)
valid.out$predicted.class<- as.factor(valid.out$predicted.class)
confusionMatrix(valid.out$predicted.class,valid.out$actuals)  
summary(residuals(ubank.glm))
plot(residuals(ubank.glm))

#install.packages("gplots")
library(gplots)
library(pROC)
library("ROCR")
pred = prediction(ubankpred,valid.df$PersonalLoan)
perf=performance(pred,"acc")
plot(perf)

perf_tpr=performance(pred,"tpr")
plot(perf_tpr)

perf_tnr=performance(pred,"tnr")
plot(perf_tnr)

r<- roc(valid.df$PersonalLoan,ubankpred)
plot.roc(r)

plot(r,print.auc=TRUE,auc.polygon=TRUE,max.auc.polygon=TRUE,auc.polygon.col='skyblue',print.thres=TRUE,grid=c(0.1,0.2),grid.col=c('green','red'))

library(gains)
gain = gains(valid.df$PersonalLoan,ubankpred)
gain

plot(c(0,gain$cume.pct.of.total*sum(valid.df$PersonalLoan))~c(0,gain$cume.obs), type = "l")
lines(c(0,sum(valid.df$PersonalLoan))~c(0,dim(valid.df)[1]),lty=2)
heights<-gain$mean.resp/mean(valid.df$PersonalLoan)
midpoint<-barplot(heights,names.arg = gain$depth,ylim = c(0,9),
                  xlab ="Percentile", ylab = "Mean Response",main = 
                    "Decile-wise lift chart")     

ubank.lm<-lm(PersonalLoan~.,data = train.df)
summary(ubank.lm)

cor(ubank)
corrplot(cor(ubank))
