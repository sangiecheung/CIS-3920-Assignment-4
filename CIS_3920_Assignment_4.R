#Sangie Cheung
#CIS 3920: Assignment 4

#Section 1
getwd()
setwd("C:/Users/Honors/Desktop")
data=read.csv("telemarketing.csv")

str(data)
data$y = as.factor(data$y)
data$contact = as.factor(data$contact)
data$housing = as.factor(data$housing)

str(data)
summary(data)

tree.data = data
set.seed(1) # for reproducibility purposes
train.index = sample(1:nrow(tree.data),nrow(tree.data)*0.80)

train = tree.data[train.index,]#train set
test = tree.data[-train.index,]#test set

summary(train$y)
summary(test$y)

#Section 2: Decision Tree
# Question 1: Create classification tree
install.packages("tree")
library(tree)
model = tree(y ~ .,data=train); summary(model)
plot(model)
text(model)

# Question 2: Pruning decision tree
best.tree = cv.tree(model,K=10) # K=10 specifying 10-fold cross validation
best.tree

x = best.tree$size
y = best.tree$dev
plot(x,y,xlab="tree size",ylab="deviance",type="b",pch=20,col="blue")

pruned.tree = prune.tree(model,best=4)
plot(pruned.tree)
text(pruned.tree)
summary(pruned.tree)

# Question 3: Show confusion matrix
pred.class = predict(pruned.tree,test,type="class") 
c.matrix = table(test$y,pred.class); c.matrix

acc = mean(test$y==pred.class)
sens.no = c.matrix[1]/(c.matrix[1]+c.matrix[3])
prec.no = c.matrix[1]/(c.matrix[1]+c.matrix[2])

data.frame(acc,sens.no,prec.no)

sens.yes = c.matrix[4]/(c.matrix[2]+c.matrix[4])
prec.yes = c.matrix[4]/(c.matrix[3]+c.matrix[4])

data.frame(acc,sens.yes,prec.yes)

#Section 3: KNN
# Data Preparation
summary(data)

# Create a function that takes a set of values, and returns normalized values
normalize = function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

# Normalize age, balance, duration, and campaign
norm.age = normalize(data$age)
norm.balance = normalize(data$balance)
norm.duration = normalize(data$duration)
norm.campaign = normalize(data$campaign)

# Change Contact variable into dummy variables
str(data)
data$cellular = 0
data$cellular[data$contact == "cellular"] = 1
data$cellular = as.factor(data$cellular)
data$telephone = 0
data$telephone [data$contact == "telephone"] = 1
data$telephone = as.factor(data$telephone)

# Change housing variables to values
data$housing = as.character(data$housing)
data$housing[data$housing=="no"] = 0 
data$housing[data$housing=="yes"] = 1 
data$housing = as.factor(data$housing)
str(data)

# Combine them into one df 
norm.data = cbind(norm.age,norm.balance,housing = data$housing,
                  norm.duration,norm.campaign, data[,8:9],y = data$y)

summary(norm.data)
str(norm.data)

# Training/Test Set
set.seed(1)
knn.train.index = sample(1:nrow(norm.data),nrow(norm.data)*0.80)

knn.train = norm.data[knn.train.index,]
knn.test = norm.data[-knn.train.index,]

# Finding the optimal K
library(class)
knn.train.x = knn.train[,1:7]
knn.test.x = knn.test[,1:7]
knn.train.cl = knn.train[,8]

# Set up 2 sets of metrics for odd K's between 3 and 5
rep = seq(3,5,2) 
rep.acc = rep
rep.sens = rep
rep.prec = rep

# Create index for 5-fold cross validation
k=5
fold = sample(1:k,nrow(knn.train.x),replace=TRUE)

iter=1 # index for rep iteration
for (K in rep) {
  
  # Space to store metrics from each iteration of k-fold cv
  kfold.acc = 1:k
  kfold.sens = 1:k
  kfold.prec = 1:k
  
  for (i in 1:k) {
    
    # data for test and training sets
    test.kfold = knn.train.x[fold==i,]
    train.kfold = knn.train.x[fold!=i,]
    
    # class labels for test and training sets
    test.cl.actual = knn.train.cl[fold==i]
    train.cl.actual = knn.train.cl[fold!=i]
    
    # make predictions on class labels for test set
    pred.class = knn(train.kfold,test.kfold,train.cl.actual,k=K)
    
    # evaluation metrics: accuracy, sensitivity, and precision (for "yes")
    c.matrix = table(test.cl.actual,pred.class)
    acc = mean(pred.class==test.cl.actual)
    sens.yes = c.matrix[4]/(c.matrix[2]+c.matrix[4])
    prec.yes = c.matrix[4]/(c.matrix[3]+c.matrix[4])
    
    # store results for each k-fold iteration
    kfold.acc[i] = acc
    kfold.sens[i] = sens.yes
    kfold.prec[i] = prec.yes
  }
  
  # store average k-fold performance for each KNN model
  rep.acc[iter] = mean(kfold.acc)
  rep.sens[iter] = mean(kfold.sens)
  rep.prec[iter] = mean(kfold.prec)
  iter=iter+1
}

# plot the results for each KNN model.
par(mfrow=c(1,3))
metric = as.data.frame(cbind(rep.acc,rep.sens,rep.prec))
color = c("blue","red","gold")
title = c("Accuracy","Sensitivity","Precision")

for (p in 1:3) {
  plot(metric[,p],type="b",col=color[p],pch=20,
       ylab="",xlab="K",main=title[p],xaxt="n")
  axis(1,at=1:2,labels=rep,las=2)
}

# Confusion matrix on KNN
pred.test.class = knn(knn.train.x,knn.test.x,knn.train.cl,k=5)
c.matrix = table(knn.test$y,pred.test.class)
c.matrix

knn.acc = mean(knn.test$y==pred.test.class)
knn.sens.yes = c.matrix[4]/(c.matrix[2]+c.matrix[4])
knn.prec.yes = c.matrix[4]/(c.matrix[3]+c.matrix[4])
as.data.frame(cbind(knn.acc,knn.sens.yes,knn.prec.yes))

