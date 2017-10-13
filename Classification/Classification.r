#install and import required packages
install.packages("caret", repos="http://cran.rstudio.com/", dependencies = TRUE)
library("rJava")
install.packages("RWeka", repos="http://cran.rstudio.com/", dependencies = TRUE)
library(caret, warn.conflicts = FALSE)
library(RWeka, warn.conflicts = FALSE)

#Training and prediction functions

#Train the training set based on C4.5 Decision trees algorithm
C45_Fit <- function(training) {
	c45_fit <- train(Continent ~., data = training, method = "J48")
	return(c45_fit)
}

#Get the predicted lables from the C4.5 trained model
C45Predict <- function(c45_fit, testing) {
	test_pred <- predict(c45_fit, newdata = testing)
	return(test_pred)
}

#Train the training set based on KNN algorithm
KNNFit <- function(training) {
	trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
	knn_fit <- train(Continent ~., data = training, method = "knn", trControl=trctrl, preProcess = c("center", "scale"), tuneLength = 10)
	print(knn_fit)
	return(knn_fit)
}

#Get the predicted lables from the KNN trained model
KNNPredict <- function (knn_fit, testing) {
	test_pred <- predict(knn_fit, newdata = testing)
	return(test_pred)
}

#Train the training set based on RIPPER Decision trees algorithm
RipperFit <- function(training) {
	ripper_fit <- train(Continent ~., data = training, method = "JRip")
}

#Get the predicted lables from the RIPPER decision trees trained model
RipperPredict <- function(svm_fit, testing) {
	test_pred <- predict(svm_fit, newdata = testing)
	return(test_pred)
}

#Train the training set based on SVM Decision trees algorithm
SVMFit <- function(training) {
	trctrl <- trainControl(method = "repeatedcv", repeats = 3)
	svm_fit <- train(Continent ~., data = training, method = "svmLinear", 
		trControl=trctrl, preProcess = c("center", "scale"))
	return(svm_fit)
}

#Get the predicted lables from the SVM trained model
SVMPredict <- function(svm_fit, testing) {
	test_pred <- predict(svm_fit, newdata = testing)
	return(test_pred)
}

#Shuffle the data based on seed value and partition the dataset into training set and testing set
divideDataset <- function(inuputfile, seedval){
	set.seed (seedval)
	continent_df <- read.csv(inuputfile)
	continent_df <- continent_df[,-2]

	#Partition data: 80% training set and 20% testing set
	intrain <- createDataPartition(continent_df$Continent, p= 0.8, list = FALSE)
	dataset <- list()
	dataset$training <- continent_df[intrain,]
	dataset$testing <- continent_df[-intrain,]
	dataset$training[["Continent"]] = factor(dataset$training[["Continent"]])
	return(dataset)
}

#Return Accuracy, precision and recall
measureAccuracy <- function(test_pred, testing, modeltype){
	#Compute confusion matrix
	mat = as.matrix(table(Actual = testing$Continent, Predicted = test_pred)) 
	print("Confusion Matrix")
	print(mat)

	#Calculate accuracy, precision, recall and F-measure
	accuracy <- sum(diag(mat)) / sum(mat)
	print(paste0("Accuracy of current iteration: ", accuracy))
	precision <- diag(mat) / rowSums(mat)
	recall <- diag(mat) / colSums(mat)
	f1 = 2 * precision * recall / (precision + recall)
	df <- data.frame(precision, recall, f1)
	print(df)
	accuracy
}

inuputfile <- "life_expectancy.csv"
knn_acc_list <- list()
svm_acc_list <- list()
c45_acc_list <- list()
ripper_acc_list <- list()
i <- 1

#Run for 5 iterations
while(i<6) {
	print(paste0("============================= Iteration: ", i))
	dataset <- divideDataset(inuputfile, i+2016)
	print(paste0("Statistics for KNN for iteration: ", i))
	knn_fit <- KNNFit(dataset$training)
	knn_pred <- KNNPredict(knn_fit, dataset$testing)
	acc <- measureAccuracy(knn_pred, dataset$testing, "KNN")
	knn_acc_list <- c(knn_acc_list, acc)

	print(paste0("Statistics for SVM for iteration: ", i))
	svm_fit <- SVMFit(dataset$training)
	svm_pred <- SVMPredict(svm_fit, dataset$testing)
	acc <-measureAccuracy(svm_pred, dataset$testing, "SVM")
	svm_acc_list <- c(svm_acc_list, acc)

	print(paste0("Statistics for C45 for iteration: ", i))
	c45_fit <- C45_Fit(dataset$training)
	c45_pred <- C45Predict(c45_fit, dataset$testing)
	acc <- measureAccuracy(c45_pred, dataset$testing, "C45")
	c45_acc_list <- c(c45_acc_list, acc)

	print(paste0("Statistics for RIPPER for iteration: ", i))
	ripper_fit <- RipperFit(dataset$training)
	ripper_pred <- RipperPredict(ripper_fit, dataset$testing)
	acc <- measureAccuracy(ripper_pred, dataset$testing, "RIPPER")
	ripper_acc_list <- c(ripper_acc_list, acc)
    i <- i + 1
}

#Average accuracies
print(paste0("Average accuracy for KNN: ", Reduce("+",knn_acc_list)/length(knn_acc_list)))
print(paste0("Average accuracy for SVM: ", Reduce("+",svm_acc_list)/length(svm_acc_list)))
print(paste0("Average accuracy for C45: ", Reduce("+",c45_acc_list)/length(c45_acc_list)))
print(paste0("Average accuracy for RIPPER: ", Reduce("+",ripper_acc_list)/length(ripper_acc_list)))









