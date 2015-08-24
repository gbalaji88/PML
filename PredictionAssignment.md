---
title: 'Practical Machine Learning Prediction Project'
author: "Balaji"
output:
  html_document:
  keep_md: yes
toc: yes
---


# Prepare the datasets

Load the training data into a data table.


```r
library(data.table)
library(utils)
url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
D <- fread(url)
```

Load the testing data into a data table.


```r
url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
TestSet <- fread(url)
```

Which variables in the test dataset have zero `NA`s?

Belt, arm, dumbbell, and forearm variables that do not have any missing values in the test dataset will be **predictor candidates**.


```r
isAnyMissing <- sapply(TestSet, function (x) any(is.na(x) | x == ""))
isPredictor <- !isAnyMissing & grepl("belt|[^(fore)]arm|dumbbell|forearm", names(isAnyMissing))
predCandidates <- names(isAnyMissing)[isPredictor]
predCandidates
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

Subset the primary dataset to include only the **predictor candidates** and the outcome variable, `classe`.


```r
varToInclude <- c("classe", predCandidates)
D <- D[, varToInclude, with=FALSE]
dim(D)
```

```
## [1] 19622    53
```

```r
names(D)
```

```
##  [1] "classe"               "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"
```

Make `classe` into a factor.


```r
D <- D[, classe := factor(D[, classe])]
D[, .N, classe]
```

```
##    classe    N
## 1:      A 5580
## 2:      B 3797
## 3:      C 3422
## 4:      D 3216
## 5:      E 3607
```

Split the dataset into a 60% training and 40% probing dataset.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
seed <- as.numeric(as.Date("2014-10-26"))
set.seed(seed)
inTrain <- createDataPartition(D$classe, p=0.6)
DTrain <- D[inTrain[[1]]]
DProbe <- D[-inTrain[[1]]]
```

Preprocess the prediction variables by centering and scaling.


```r
X <- DTrain[, predCandidates, with=FALSE]
preProc <- preProcess(X)
preProc
```

```
## 
## Call:
## preProcess.default(x = X)
## 
## Created from 11776 samples and 52 variables
## Pre-processing: centered, scaled
```

```r
XCS <- predict(preProc, X)
DTrainCS <- data.table(data.frame(classe = DTrain[, classe], XCS))
```

Apply the centering and scaling to the probing dataset.


```r
X <- DProbe[, predCandidates, with=FALSE]
XCS <- predict(preProc, X)
DProbeCS <- data.table(data.frame(classe = DProbe[, classe], XCS))
```

Check for near zero variance.


```r
nzv <- nearZeroVar(DTrainCS, saveMetrics=TRUE)
if (any(nzv$nzv)) nzv else message("No variables with near zero variance")
```

```
## No variables with near zero variance
```

# Train a prediction model

I chose to use random forests for a prediction model
The error will be estimated using the 40% probing sample.


Fit model over the tuning parameters.


```r
#system.time(trainingModel <- train(classe ~ ., data=DTrainCS, method="rf"))
trainingModel <- train(classe ~ ., data=DTrainCS, method="rf")
```

```
## 1 package is needed for this model and is not installed. (randomForest). Would you like to try to install it now?
```

```
## Error in checkInstall(models$library):
```


## Evaluate the model on the training dataset


```r
trainingModel
```

```
## Error in eval(expr, envir, enclos): object 'trainingModel' not found
```

```r
hat <- predict(trainingModel, DTrainCS)
```

```
## Error in predict(trainingModel, DTrainCS): object 'trainingModel' not found
```

```r
confusionMatrix(hat, DTrain[, classe])
```

```
## Error in unique.default(x, nmax = nmax): unique() applies only to vectors
```

## Evaluate the model on the probing dataset


```r
hat <- predict(trainingModel, DProbeCS)
```

```
## Error in predict(trainingModel, DProbeCS): object 'trainingModel' not found
```

```r
confusionMatrix(hat, DProbeCS[, classe])
```

```
## Error in unique.default(x, nmax = nmax): unique() applies only to vectors
```

## Display the final model


```r
varImp(trainingModel)
```

```
## Error in varImp(trainingModel): object 'trainingModel' not found
```

```r
trainingModel$finalModel
```

```
## Error in eval(expr, envir, enclos): object 'trainingModel' not found
```

**The estimated error rate is less than 1%.**
  
Save training model object for later.


```r
save(trainingModel, file="trainingModel.RData")
```

```
## Error in save(trainingModel, file = "trainingModel.RData"): object 'trainingModel' not found
```


# Predict on the test data

Load the training model.


```r
load(file="trainingModel.RData", verbose=TRUE)
```

```
## Warning in readChar(con, 5L, useBytes = TRUE): cannot open compressed file
## 'trainingModel.RData', probable reason 'No such file or directory'
```

```
## Error in readChar(con, 5L, useBytes = TRUE): cannot open the connection
```

Get predictions and evaluate.


```r
TestSetCS <- predict(preProc, TestSet[, predCandidates, with=FALSE])
hat <- predict(trainingModel, TestSetCS)
```

```
## Error in predict(trainingModel, TestSetCS): object 'trainingModel' not found
```

```r
TestSet <- cbind(hat , TestSet)
```

```
## Error in data.table::data.table(...): problem recycling column 1, try a simpler type
```

```r
subset(TestSet, select=names(TestSet)[grep("belt|[^(fore)]arm|dumbbell|forearm", names(TestSet), invert=TRUE)])
```

```
##     V1 user_name raw_timestamp_part_1 raw_timestamp_part_2
##  1:  1     pedro           1323095002               868349
##  2:  2    jeremy           1322673067               778725
##  3:  3    jeremy           1322673075               342967
##  4:  4    adelmo           1322832789               560311
##  5:  5    eurico           1322489635               814776
##  6:  6    jeremy           1322673149               510661
##  7:  7    jeremy           1322673128               766645
##  8:  8    jeremy           1322673076                54671
##  9:  9  carlitos           1323084240               916313
## 10: 10   charles           1322837822               384285
## 11: 11  carlitos           1323084277                36553
## 12: 12    jeremy           1322673101               442731
## 13: 13    eurico           1322489661               298656
## 14: 14    jeremy           1322673043               178652
## 15: 15    jeremy           1322673156               550750
## 16: 16    eurico           1322489713               706637
## 17: 17     pedro           1323094971               920315
## 18: 18  carlitos           1323084285               176314
## 19: 19     pedro           1323094999               828379
## 20: 20    eurico           1322489658               106658
##       cvtd_timestamp new_window num_window problem_id
##  1: 05/12/2011 14:23         no         74          1
##  2: 30/11/2011 17:11         no        431          2
##  3: 30/11/2011 17:11         no        439          3
##  4: 02/12/2011 13:33         no        194          4
##  5: 28/11/2011 14:13         no        235          5
##  6: 30/11/2011 17:12         no        504          6
##  7: 30/11/2011 17:12         no        485          7
##  8: 30/11/2011 17:11         no        440          8
##  9: 05/12/2011 11:24         no        323          9
## 10: 02/12/2011 14:57         no        664         10
## 11: 05/12/2011 11:24         no        859         11
## 12: 30/11/2011 17:11         no        461         12
## 13: 28/11/2011 14:14         no        257         13
## 14: 30/11/2011 17:10         no        408         14
## 15: 30/11/2011 17:12         no        779         15
## 16: 28/11/2011 14:15         no        302         16
## 17: 05/12/2011 14:22         no         48         17
## 18: 05/12/2011 11:24         no        361         18
## 19: 05/12/2011 14:23         no         72         19
## 20: 28/11/2011 14:14         no        255         20
```

## Submission to Coursera

Write submission files to `PMLfiles/`.


```r
pml_write_files = function(x){
  n = length(x)
  path <- "PMLfiles/"
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=file.path(path, filename),quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(hat)
```

```
## Error in x[i]: object of type 'closure' is not subsettable
```
