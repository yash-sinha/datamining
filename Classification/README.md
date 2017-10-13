The project uses Classification.r for classification of countries into different Continents. The dataset includes 223 countries.

To run the program:
1. Make sure the data file "life_expectancy.csv" is in the same directory as the Rscript Classification.r
2. Run: Rscript Classification.r

Following packages are installed using the Rscript (these have been already written int R script and we need not do anything extra):
install.packages("caret", repos="http://cran.rstudio.com/", dependencies = TRUE)
install.packages("RWeka", repos="http://cran.rstudio.com/", dependencies = TRUE)

The following are the functions used in the file Classification.r file:

1. C45_Fit <- function(training) 
2. C45Predict <- function(c45_fit, testing)
3. KNNFit <- function(training)
4. KNNPredict <- function (knn_fit, testing)
5. RipperFit <- function(training)
6. RipperPredict <- function(svm_fit, testing) 
7. SVMFit <- function(training)
8. SVMPredict <- function(svm_fit, testing)
9. divideDataset <- function(inuputfile, seedval)
10. measureAccuracy <- function(test_pred, testing, modeltype)

The 'fit' functions return the corresponding trained models. These take the training set as parameter.
The predict functions return the predicted labels. These take the trained model and the testing set as parameter.
The divideDataset function takes the input file name and a seed value to shuffle the data. This ensures random data in each iteration. This function
	partions the dataset into 80% training set and 20% testing set. It then returns the dataset.
The measureAccuracy function takes the predicted labels, testing set and modeltype as parameters. It then creates a confusion matrix and calculates 
	accuracy, precision and recall. It returns the accuracy which is then appended to accuracy list of each of the algorithms. Finally, average of 
	these accuracies is calculated to print the average accuracy.

Interpreting outputs:
The dataset is shuffled for 5 iterations. For each of these iterations, the confusion matrix, accuracy, precision, recall and f-measure for each class are printed. In the case of
KNN, the accuracies of the 10 Ks are printed out. Finally, the average accuracy is printed out for each model.
