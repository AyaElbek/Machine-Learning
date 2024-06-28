# Bank Campaign Customer Classification
# Authors: Elbekova Aidai, Bethelhem Samson Gebreegziabhier, Joshua Adu
# Supervisors: Prof. Dr. André Hanelt, Steven Görlich, M. Sc.

# Load necessary libraries
library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(nnet)

# Load the dataset
dataset <- read.csv("bank-full.csv", sep=";")

# Data Preprocessing

# Finding Missing Values
missing_values <- sapply(dataset, function(x) sum(is.na(x)))
print(missing_values)

# Encoding Categorical Data
dataset$job <- as.factor(dataset$job)
dataset$marital <- as.factor(dataset$marital)
dataset$education <- as.factor(dataset$education)
dataset$default <- as.factor(dataset$default)
dataset$housing <- as.factor(dataset$housing)
dataset$loan <- as.factor(dataset$loan)
dataset$contact <- as.factor(dataset$contact)
dataset$month <- factor(dataset$month, levels = c("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
dataset$day_of_week <- factor(dataset$day_of_week, levels = c("mon", "tue", "wed", "thu", "fri"))
dataset$poutcome <- as.factor(dataset$poutcome)
dataset$y <- as.factor(dataset$y)

# One-hot encoding
dataset <- dummyVars(" ~ .", data = dataset)
dataset <- data.frame(predict(dataset, newdata = dataset))

# Splitting the Dataset
set.seed(123)
trainIndex <- createDataPartition(dataset$y, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- dataset[ trainIndex,]
test_data  <- dataset[-trainIndex,]

# Feature Scaling
preProcValues <- preProcess(train_data, method = c("center", "scale"))
train_data <- predict(preProcValues, train_data)
test_data <- predict(preProcValues, test_data)

# Random Oversampling
train_data <- upSample(x = train_data, y = train_data$y)

# Model Description

# Logistic Regression
model_lr <- train(y ~ ., data = train_data, method = "glm", family = "binomial")
predictions_lr <- predict(model_lr, test_data)
confusionMatrix(predictions_lr, test_data$y)

# Random Forest
model_rf <- train(y ~ ., data = train_data, method = "rf")
predictions_rf <- predict(model_rf, test_data)
confusionMatrix(predictions_rf, test_data$y)

# Support Vector Machine
model_svm <- train(y ~ ., data = train_data, method = "svmRadial")
predictions_svm <- predict(model_svm, test_data)
confusionMatrix(predictions_svm, test_data$y)

# Neural Network
model_nn <- train(y ~ ., data = train_data, method = "nnet", linout = TRUE)
predictions_nn <- predict(model_nn, test_data)
confusionMatrix(predictions_nn, test_data$y)

# K-Nearest Neighbors
model_knn <- train(y ~ ., data = train_data, method = "knn")
predictions_knn <- predict(model_knn, test_data)
confusionMatrix(predictions_knn, test_data$y)

# Ada Boost
model_ada <- train(y ~ ., data = train_data, method = "adaboost")
predictions_ada <- predict(model_ada, test_data)
confusionMatrix(predictions_ada, test_data$y)

# Naive Bayes
model_nb <- train(y ~ ., data = train_data, method = "nb")
predictions_nb <- predict(model_nb, test_data)
confusionMatrix(predictions_nb, test_data$y)

# Decision Trees
model_dt <- train(y ~ ., data = train_data, method = "rpart")
predictions_dt <- predict(model_dt, test_data)
confusionMatrix(predictions_dt, test_data$y)

# Data Optimization
# Hyperparameter tuning and k-fold validation (example with Random Forest)

control <- trainControl(method="cv", number=10)
tunegrid <- expand.grid(.mtry=c(1:15))
model_rf_opt <- train(y ~ ., data=train_data, method="rf", trControl=control, tuneGrid=tunegrid)
predictions_rf_opt <- predict(model_rf_opt, test_data)
confusionMatrix(predictions_rf_opt, test_data$y)

# Print the optimized Random Forest model details
print(model_rf_opt)

# Save the models for later use
saveRDS(model_rf, "model_rf.rds")
saveRDS(model_rf_opt, "model_rf_opt.rds")
