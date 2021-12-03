#############################################
#                                           #
# Author:     Dustin Berry                  #
# Date:       18 August 2020                #
# Subject:    Final Project                 #
# Class:      BDAT 640                      #
# Section:    01W                           #         
# Instructor: Kubrom Teka                   #
# File Name:  FinalProject_Berry_Dustin.R   #
#                                           # 
#############################################


########################
# 1.  Data Preparation #
########################

#     a.  Load the dataset insurance.csv into memory.
library(readr)
insurance <- read.csv("Maryville/Predictive Modeling/insurance.csv")
View(insurance)

insurance$sex <- factor(insurance$sex)
insurance$smoker <- factor(insurance$smoker)
insurance$region <- factor(insurance$region)

#     b.  In the data frame, transform the variable charges by setting
#         insurance$charges = log(insurance$charges). Do not transform
#         it outside of the data frame.
insurance$charges <-  log(insurance$charges)

#     c.  Using the data set from 1.b, use the model.matrix() function
#         to create another data set that uses dummy variables in place
#         of categorical variables. Verify that the first column only has
#         ones (1) as values, and then discard the column only after
#         verifying it has only ones as values.
mm.ins <- model.matrix(~., data=insurance)
mm.ins

mm.ins <- mm.ins[,-1]
mm.ins

#     d.  Use the sample() function with set.seed equal to 1 to generate
#         row indexes for your training and tests sets, with 2/3 of the
#         row indexes for your training set and 1/3 for your test set. Do
#         not use any method other than the sample() function for
#         splitting your data.
set.seed(1)
index <- sample(1:nrow(insurance), 2*nrow(insurance)/3)

#     e.  Create a training and test data set from the data set created in
#         1.b using the training and test row indexes created in 1.d.
#         Unless otherwise stated, only use the training and test
#         data sets created in this step.
train <- insurance[index, ]
test <- insurance[-index, ]

#     f.  Create a training and test data set from data set created in 1.c
#         using the training and test row indexes created in 1.d
mm.train <- mm.ins[index, ]
mm.test <- mm.ins[-index, ]

#################################################
# 2.  Build a multiple linear regression model. #
#################################################

#     a.  Perform multiple linear regression with charges as the
#         response and the predictors are age, sex, bmi, children,
#         smoker, and region. Print out the results using the
#         summary() function. Use the training data set created in
#         step 1.e to train your model.

lm.fit = lm(charges ~ age + sex + bmi + children + smoker + region, data = train)
summary(lm.fit)

#     b.  Is there a relationship between the predictors and the
#         response?
#With a p-value of 2.2e-16 there is a clear relationship between the 
#predictors and the response.

#     c.  Does sex have a statistically significant relationship to the
#         response?
#Yes, the p-value is less than 5% (.027847).

#     d.  Perform best subset selection using the stepAIC() function
#         from the MASS library, choose best model based on AIC. For
#         the "direction" parameter in the stepAIC() method, set
#         direciton="backward".
library(MASS)

fit.ins <- glm(charges ~ ., data=train)
fit.best <- stepAIC(fit.ins, scope=list(lower= ~ 1), direction="backward")
#AIC=1019.4

summary(fit.best)

#     e.  Compute the test error of the best model in #2d based on AIC
#         using LOOCV using trainControl() and train() from the caret
#         library. Report the MSE by squaring the reported RMSE.
library(caret)

train_control <- trainControl(method="LOOCV")
fit2e <- train(charges ~ ., data=train, trcontrol=train_control, method="lm")
fit2e

(0.4307522)^2
#MSE = 0.1855475

#     f.  Calculate the test error of the best model in #2d based on AIC
#         using 10-fold Cross-Validation. Use train and trainControl
#         from the caret library. Refer to model selected in #2d based
#         on AIC. Report the MSE.

train_control2 <- trainControl(method="CV", number=10)
fit2f <- train(charges ~ ., data=train, trcontrol=train_control2, method="lm")
fit2f

(0.4422772)^2
#MSE = 0.1956091

#     g.  Calculate and report the test MSE using the best model from 
#         2.d and the test data set from step 1.e.
pred2g <- predict(fit.best, newdata=test)

MSE2 <- mean((test[,"charges"] - pred2g)^2)
MSE2
#MSE = 0.231291

#     h.  Compare the test MSE calculated in step 2.f using 10-fold
#         cross-validation with the test MSE calculated in step 2.g.
#         How similar are they?
#2f MSE = 0.1956091
#2g MSE = 0.231291
#I would consider this a fair amount of variance, and the model in step
#2f would be the better model.


######################################
# 3.  Build a regression tree model. #
######################################

#     a.  Build a regression tree model using function tree(), where
#         charges is the response and the predictors are age, sex, bmi,
#         children, smoker, and region.
library(tree)

treemodel <- formula(charges ~ age + sex + bmi + children + smoker + region)
fit.tree <- tree(treemodel, data=train)

#     b.  Find the optimal tree by using cross-validation and display
#         the results in a graphic. Report the best size.
cv.tree.results <- cv.tree(fit.tree)

plot(cv.tree.results, type="b")
#The optimal tree would have five terminal nodes. 

#c.  Justify the number you picked for the optimal tree with regard to the
#principle of variance-bias trade-off.
#The optimal tree would have five terminal nodes as it has a good balance of
#small deviance and size.

#     d.  Prune the tree using the optimal size found in 3.b.
prune.tree.model <- prune.tree(fit.tree, best = 5)

#     e.  Plot the best tree model and give labels.
plot(prune.tree.model)
text(prune.tree.model, pretty=0)

#     f.  Calculate the test MSE for the best model.
pred <- predict(prune.tree.model, newdata=test)

MSE3 <- mean((test[,"charges"] - pred)^2)
MSE3
#MSE = 0.218803

####################################
# 4.  Build a random forest model. #
####################################

#     a.  Build a random forest model using function randomForest(),
#         where charges is the response and the predictors are age, sex,
#         bmi, children, smoker, and region.
library(randomForest)

fit.rf <- randomForest(charges ~ age + sex + bmi + children + smoker + 
                         region, data=train, importance=T)

#     b.  Compute the test error using the test data set.
pred.rf <- predict(fit.rf, newdata=test, type="response")
MSE.rf  <- mean((test[,"charges"] - pred.rf)^2)
MSE.rf
#MSE = 0.178367

#     c.  Extract variable importance measure using the importance()
#         function.
importance(fit.rf)

#     d.  Plot the variable importance using the function, varImpPlot().
#         Which are the top 3 important predictors in this model?
varImpPlot(fit.rf)
#The top three important predictors are smoker, age, and children.

############################################
# 5.  Build a support vector machine model #
############################################

#     a.  The response is charges and the predictors are age, sex, bmi,
#         children, smoker, and region. Please use the svm() function
#         with radial kernel and gamma=5 and cost = 50.
library(e1071)

svm.fit = svm(charges ~ age + sex + bmi + children + smoker + region, 
              data=train, kernel="radial", gamma=5, cost=50)
summary(svm.fit)

#     b.  Perform a grid search to find the best model with potential
#         cost: 1, 10, 50, 100 and potential gamma: 1,3 and 5 and
#         potential kernel: "linear","radial" and "sigmoid". And use 
#         the training set created in step 1.e.

#Discrepancy in the example file and Canvas: "polynomial" kernal is not listed
#in Canvas so I left it out.

tune.out = tune(svm, charges ~ age + sex + bmi + children + smoker + region, 
                data=train, kernel=c("linear", "radial", "sigmoid"), 
                ranges=list(cost=c(1, 10, 50, 100), gamma=c(1, 3, 5)))

#     c.  Print out the model results. What are the best model
#         parameters?
summary(tune.out)
#The best model parameters have a cost of 1.  The gamma doesn't impact the
#performance of the model in either a positive or negative manner at any point
#given the same cost, so I would go with a gamma of 1 for simplicity.  The best
#performance error is 0.1949692.

#     d.  Forecast charges using the test dataset and the best model
#         found in c).
pred.svm = predict(tune.out$best.model, newdata=test)

#     e.  Compute the MSE (Mean Squared Error) on the test data.
MSE.svm  <- mean((test[,"charges"] - pred.svm)^2)
MSE.svm
#MSE = 0.257127

#############################################
# 6.  Perform the k-means cluster analysis. #
#############################################

#Discrepancies between the example file and Canvas.  I copied the questions from
#Canvas.

#     a.  Remove the sex, smoker, and region, since they are not numerical
#         values.
data.6 <- insurance[,-c(2,5,6)]
data.6

#     b.  Determine the optimal number of clusters. Justify your answer. It may
#         take longer running time since it uses a large dataset.
library(cluster)
library(factoextra)
fviz_nbclust(data.6, kmeans, method = "gap_stat")
#The optimal number of clusters is 2 based on the fviz_nbclust function using
#the gap_stats method.

#     c.  Perform k-means clustering using the 3 clusters.
km.res <- kmeans(data.6, 3, nstart = 25)

#     d.  Visualize the clusters in different colors.
fviz_cluster(km.res, data = data.6)

######################################
# 7.  Build a neural networks model. #
######################################

#Discrepancies between the example file and Canvas.  I copied the questions from
#Canvas.

#     a.  Remove the sex, smoker, and region, since they are not numerical
#     values.
data.7 <- insurance[,-c(2,5,6)]
data.7

#     b.  Standardize the inputs using the scale() function.
scaled.data.7 = scale(data.7)

#     c.  Convert the standardized inputs to a data frame using the
#     as.data.frame() function.
scaled.data.7 = as.data.frame(scaled.data.7)

#     d.  Split the dataset into a training set containing 80% of the original
#     data and the test set containing the remaining 20%.
index.nn <- sample(1:nrow(scaled.data.7),0.80*nrow(scaled.data.7))
train.nn <- scaled.data.7[index.nn,]
test.nn <- scaled.data.7[-index.nn,]

#     e.  The response is charges and the predictors are age, bmi, and children.
#     Please use 1 hidden layer with 1 neuron.
library(neuralnet)
nn.model <- neuralnet(charges ~ age + bmi + children, data=train.nn, hidden=1)
#line 299 gives the warning of "Algorithm did not converge in 1 of 1
#repetition(s) within the stepmax." when the entire page of code is executed at
#once, but runs fine when I run just that line

#     f.  Plot the neural networks.
plot(nn.model)
#line 305 gives the error of "Error in plot.nn(nn.model) : weights were not
#calculated" when the entire page of code is executed at once, but runs fine
#when I run just that line

#     g.  Forecast the charges in the test dataset.
predict.nn = compute(nn.model, test.nn[, c("age", "bmi", "children")])
#line 311 gives the error of "Error in cbind(1, pred) %*%
#weights[[num_hidden_layers + 1]] : requires numeric/complex matrix/vector
#arguments" when the entire page of code is executed at once, but runs fine when
#I run just that line

#     h.  Get the observed charges of the test dataset.
observe.test <- test.nn$charges

#     i.  Compute test error (MSE).
MSE.nn <- mean((observe.test - predict.nn$net.result)^2)
MSE.nn
#MSE = 0.6615006

################################
# 8.  Putting it all together. #
################################

#     a.  For predicting insurance charges, your supervisor asks you to
#         choose the best model among the multiple regression,
#         regression tree, random forest, support vector machine, and
#         neural network models. Compare the test MSEs of the models
#         generated in steps 2.g, 3.f, 4.b, 5.e, and 7.i. Display the names
#         for these types of these models, using these labels:
#         "Multiple Linear Regression", "Regression Tree", "Random Forest", 
#         "Support Vector Machine", and "Neural Network" and their
#         corresponding test MSEs in a data.frame. Label the column in your
#         data frame with the labels as "Model.Type", and label the column
#         with the test MSEs as "Test.MSE" and round the data in this
#         column to 4 decimal places. Present the formatted data to your
#         supervisor and recommend which model is best and why.

compare.MSE.df <- data.frame("Model.Type" = c("Multiple Linear Regression", 
                  "Regression Tree", "Random Forest", "Support Vector Machine",
                  "Neural Network"), "Test.MSE" = c(0.2313, 0.2188, 0.1784,
                                                    0.2571, 0.6615))
print(compare.MSE.df)
#I recommend the Random Forest Model as it produces the least error of the given
#models.

#     b.  Another supervisor from the sales department has requested
#         your help to create a predictive model that his sales
#         representatives can use to explain to clients what the potential
#         costs could be for different kinds of customers, and they need
#         an easy and visual way of explaining it. What model would
#         you recommend, and what are the benefits and disadvantages
#         of your recommended model compared to other models?

#My recommendation would be the Regression Tree model as it's the easiest to
#visualize and explain while maintaining a lower error rate when compared to the
#other models with the exception of the Random Forest model.  Although the
#Random Forest model has a lower error rate, is very technical and does not lend
#itself to easy explanation for those without prior training.

#     c.  The supervisor from the sales department likes your regression
#         tree model. But she says that the sales people say the numbers
#         in it are way too low and suggests that maybe the numbers
#         on the leaf nodes predicting charges are log transformations
#         of the actual charges. You realize that in step 1.b of this
#         project that you had indeed transformed charges using the log
#         function. And now you realize that you need to reverse the
#         transformation in your final output. The solution you have
#         is to reverse the log transformation of the variables in 
#         the regression tree model you created and redisplay the result.
#         Follow these steps:
#
#         i.   Copy your pruned tree model to a new variable.
prune.tree.model.copy <- prune.tree.model

#         ii.  In your new variable, find the data.frame named
#              "frame" and reverse the log transformation on the
#              data.frame column yval using the exp() function.
#              (If the copy of your pruned tree model is named 
#              copy_of_my_pruned_tree, then the data frame is
#              accessed as copy_of_my_pruned_tree$frame, and it
#              works just like a normal data frame.).
prune.tree.model.copy$frame$yval <- exp(prune.tree.model.copy$frame$yval)

#         iii. After you reverse the log transform on the yval
#              column, then replot the tree with labels.
plot(prune.tree.model.copy)
text(prune.tree.model.copy, pretty = 0)

