# HAR.RMD
Kamal Rautela  
January 11, 2017  



Summary
----------------------------------------------------

There was a research conducted on the quality of executing an activity during weight Lifting Excercise. For this research model and sensor approach was used. Under this apporach, data (WLE Dataset) was collected from 6 participants who were asked to perform barbell lifts correctly and incorrectly. In this project, I have taken that data and identified a prediction model that gives predictions with an accuracy of 95%. The prediction model selected was "Random Forest" (RF). The other two models that were tested were Linear Discriminant Analysis (LDA) and Regression Classification. These models provided prediction models with relatively lower accuracy as compared to Random Forest.


Read and load the data. 
------------------------------------------------------


```r
trainingData<-read.csv("./pml-training.csv",header=TRUE)
testingData<-read.csv("./pml-testing.csv",header=TRUE)
```


Clean the data
-------------------------------------------------------


```r
##Identify the variables which have less than 20% NA values and keep them

nonNAcols<-colnames(trainingData)[colMeans(is.na(trainingData)) < .2]

trainingData<-trainingData[,nonNAcols]

##This left us with 93 variables

##Identify the variables which have less than 20% empty values and keep them

nonEmptycols<-colnames(trainingData)[colMeans(trainingData=="") < .2]

trainingData<-trainingData[,nonEmptycols]

##This left us with 60 variables

##Identify the columsn which have near zero variance 
## and remove them

library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
trainingData<-trainingData[,-nearZeroVar(trainingData)]

##This left us with 59 variables

##The first 6 columns are row identifier, username, 3 time stamps columns and window number
##These can be used a predictors so eliminate these as well

trainingData<-trainingData[,-(1:6)]

##This left us with 53 variables 
```


Create test data and validation data set partitions from training data
------------------------------------------------------


```r
library(caret)

set.seed(1)

inTrain<-createDataPartition(y=trainingData$classe,p=.60,list=FALSE)

training<-trainingData[inTrain,]
testing<-trainingData [-inTrain,]
```




Explore the data / Exploratory Analysis
----------------------------------------------

Steps:

1. Plots to identify any trends - too many variable to plot!
2. Identify coorelated columns
3. Identify preprocessing that may be required



```r
dim(training)
```

```
## [1] 11776    53
```

```r
M<-abs(cor(training[,-53]))
diag(M)<-0
length(which(M>.8,arr.ind=T))
```

```
## [1] 76
```

```r
##This shows that there are 76 pairs of columns which are highly coorelated
##Therefore it makes sense to use Principal Component Analysis in order to reduce variance
##This will have a negative impact on bias but will help reduce variance in the prediction model
```


Build model 
------------------------

We can try to fit the following models:

1. Classification trees
2. Random Forest
3. Linear Discriminant Analysis

Before we train our models, we are going to define the following :

1. metric - Since our outcome is a factor variable, we will choose "Accuracy"" as our criteria for model selection. We will measure Accurarcy with function - confustionMatrix function

2. preProcess - Principal Component Analysis ("pca") where appropiate in order to reduce covariates in the favor of saving run time and/or reducing variance.  

3. trainControl - We will use cross-validation in order to reduce bias where appropiate


Model1 - Classification with tree
----------------------------------


```r
set.seed(2)

##Fit the model without preprocessing and cross-validation

modelFitTree<- train(classe~.,data=training,method="rpart")
```

```
## Loading required package: rpart
```

```r
finModTree<-modelFitTree$finalModel

library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
fancyRpartPlot(finModTree)
```

![](HAR_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

```r
## Find out of sample error rate

confusionMatrix(predict(modelFitTree,newdata = testing),testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2019  626  626  584  210
##          B   36  517   41  234  198
##          C  172  375  701  468  387
##          D    0    0    0    0    0
##          E    5    0    0    0  647
## 
## Overall Statistics
##                                           
##                Accuracy : 0.495           
##                  95% CI : (0.4839, 0.5062)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3402          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9046  0.34058  0.51243   0.0000  0.44868
## Specificity            0.6356  0.91956  0.78358   1.0000  0.99922
## Pos Pred Value         0.4967  0.50390  0.33333      NaN  0.99233
## Neg Pred Value         0.9437  0.85323  0.88386   0.8361  0.88949
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2573  0.06589  0.08934   0.0000  0.08246
## Detection Prevalence   0.5181  0.13077  0.26803   0.0000  0.08310
## Balanced Accuracy      0.7701  0.63007  0.64800   0.5000  0.72395
```

```r
## This gives an accuracy of 49.96% which is not acceptable


##Now lets try fitting the model with cross-validation to see if improve accuracy


train.control <- trainControl (method="cv", number=4, allowParallel = TRUE)

modelFitTree2<- train(classe~.,data=training,method="rpart",trControl=train.control)
                    
finModTree2<-modelFitTree2$finalModel

print(finModTree2)
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10783 7444 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 948    7 A (0.99 0.0074 0 0 0) *
##      5) pitch_forearm>=-33.95 9835 7437 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 439.5 8332 5979 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 124.5 5236 3112 A (0.41 0.18 0.18 0.17 0.06) *
##         21) roll_forearm>=124.5 3096 2076 C (0.074 0.18 0.33 0.23 0.19) *
##       11) magnet_dumbbell_y>=439.5 1503  734 B (0.03 0.51 0.045 0.22 0.19) *
##    3) roll_belt>=130.5 993    9 E (0.0091 0 0 0 0.99) *
```

```r
## Find out of sample error rate

confusionMatrix(predict(modelFitTree2,newdata = testing),testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2019  626  626  584  210
##          B   36  517   41  234  198
##          C  172  375  701  468  387
##          D    0    0    0    0    0
##          E    5    0    0    0  647
## 
## Overall Statistics
##                                           
##                Accuracy : 0.495           
##                  95% CI : (0.4839, 0.5062)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3402          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9046  0.34058  0.51243   0.0000  0.44868
## Specificity            0.6356  0.91956  0.78358   1.0000  0.99922
## Pos Pred Value         0.4967  0.50390  0.33333      NaN  0.99233
## Neg Pred Value         0.9437  0.85323  0.88386   0.8361  0.88949
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2573  0.06589  0.08934   0.0000  0.08246
## Detection Prevalence   0.5181  0.13077  0.26803   0.0000  0.08310
## Balanced Accuracy      0.7701  0.63007  0.64800   0.5000  0.72395
```

This gives an accuracy of 49.5% which is even worse (not acceptable again)


Model2 - Random Forest
----------------------------------


```r
gc()
```

```
##            used  (Mb) gc trigger  (Mb) max used  (Mb)
## Ncells  1897192 101.4    3205452 171.2  2762985 147.6
## Vcells 11538225  88.1   21536919 164.4 21535855 164.4
```

```r
set.seed(1)

preProc<-preProcess(training[,-53],method="pca", thresh=.8)

trainPC<-predict(preProc,training[,-53])

##preprocessing with PCA reduces covariates from 52 to 12
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
trainPC<-mutate(trainPC, classe=training$classe)

train.control <- trainControl (method="cv", number=4, allowParallel = TRUE)

modelFitRF <- train(classe ~ ., data=trainPC, method="rf", 
                    trControl = train.control)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
finModRF<-modelFitRF$finalModel

print(finModRF)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 4.75%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3251   20   40   30    7  0.02897252
## B   71 2122   57   14   15  0.06888986
## C   17   37 1966   25    9  0.04284323
## D   17   12  113 1784    4  0.07564767
## E    4   28   19   20 2094  0.03279446
```

```r
## Find out of sample error rate

testPC<- predict(preProc,testing[,-53])
confusionMatrix(predict(modelFitRF,newdata = testPC),testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2149   49    8   14    8
##          B   19 1412   18    7    8
##          C   33   43 1322   72   14
##          D   20    5   17 1191   15
##          E   11    9    3    2 1397
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9522          
##                  95% CI : (0.9473, 0.9568)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9396          
##  Mcnemar's Test P-Value : 2.466e-15       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9628   0.9302   0.9664   0.9261   0.9688
## Specificity            0.9859   0.9918   0.9750   0.9913   0.9961
## Pos Pred Value         0.9645   0.9645   0.8908   0.9543   0.9824
## Neg Pred Value         0.9852   0.9834   0.9928   0.9856   0.9930
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2739   0.1800   0.1685   0.1518   0.1781
## Detection Prevalence   0.2840   0.1866   0.1891   0.1591   0.1812
## Balanced Accuracy      0.9744   0.9610   0.9707   0.9587   0.9824
```

This gives an accuracy of 95.09% which is pretty good.



Model3 - Linear Discriminant Analysis
---------------------------------------------


```r
gc()
```

```
##            used (Mb) gc trigger  (Mb) max used  (Mb)
## Ncells  1965039  105    3205452 171.2  3100733 165.6
## Vcells 18350035  140   75732364 577.8 94605263 721.8
```

```r
set.seed(3)

train.control <- trainControl (method="cv", number=4, allowParallel = TRUE)

modelFitLDA <- train(classe ~ ., data=training, method="lda")
```

```
## Loading required package: MASS
```

```
## 
## Attaching package: 'MASS'
```

```
## The following object is masked from 'package:dplyr':
## 
##     select
```

```r
finModLDA<-modelFitLDA$finalModel

print(finModLDA)
```

```
## Call:
## lda(x, grouping = y)
## 
## Prior probabilities of groups:
##         A         B         C         D         E 
## 0.2843071 0.1935292 0.1744226 0.1638927 0.1838485 
## 
## Group means:
##   roll_belt pitch_belt   yaw_belt total_accel_belt gyros_belt_x
## A  59.76690  0.3024223 -11.377345         10.72879 -0.007004182
## B  63.93125  0.1993857 -15.369833         10.97674 -0.001110136
## C  65.73020 -1.4994401  -5.208466         11.32376 -0.014961052
## D  60.35336  1.5442124 -17.744902         11.19067 -0.011419689
## E  74.28719  0.4970947  -5.332734         12.68176  0.008688222
##   gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## A   0.04050179   -0.1226523    -6.222521     29.13381    -63.36589
## B   0.04284335   -0.1311716    -5.186047     31.47477    -71.73234
## C   0.04024343   -0.1361198    -3.399221     31.38023    -72.21081
## D   0.03708808   -0.1329534    -8.040933     30.16528    -68.17409
## E   0.03899769   -0.1225543    -4.467436     28.88961    -91.14411
##   magnet_belt_x magnet_belt_y magnet_belt_z   roll_arm  pitch_arm
## A      58.21117      602.0753     -338.2536 -0.9777748   3.779005
## B      48.38394      599.3541     -336.7240 31.4082361  -5.913659
## C      59.08423      600.4172     -336.8082 23.6279844  -1.500657
## D      49.32124      594.8016     -340.0031 21.8463212 -10.256788
## E      63.94365      568.8988     -375.8600 20.1901016 -12.541686
##      yaw_arm total_accel_arm  gyros_arm_x gyros_arm_y gyros_arm_z
## A -12.806637        27.43967  0.043207885  -0.2257348   0.2685245
## B   8.736999        26.72663 -0.009113646  -0.2589294   0.2637867
## C   2.273101        24.24051  0.128101266  -0.2790944   0.2850195
## D   5.365487        23.23212  0.059569948  -0.2555026   0.2683109
## E   0.142485        24.73995  0.038106236  -0.2837506   0.2836582
##   accel_arm_x accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y
## A  -132.12784    46.76404   -74.78345    -20.28196    236.38172
## B   -46.31461    26.62484   -96.80474    228.89908    130.76349
## C   -77.74537    38.90312   -56.24927    155.97858    189.41723
## D    14.20104    26.22280   -46.48031    397.73938     97.75907
## E   -18.68406    16.08545   -77.65820    322.81155     83.85589
##   magnet_arm_z roll_dumbbell pitch_dumbbell yaw_dumbbell
## A     411.7392      22.29995     -18.793979     1.369078
## B     200.0360      36.34516       3.694592    14.413999
## C     359.2327     -11.82623     -25.627212   -18.363047
## D     301.5523      49.86084      -3.518552    -0.685985
## E     213.8891      26.66955      -7.242171     5.049431
##   total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z
## A             14.63441        0.1089845      0.037255078      -0.05764636
## B             14.40193        0.1752523      0.004238701      -0.14283019
## C             13.35346        0.1972395      0.058588121      -0.15776047
## D             11.49845        0.2057358      0.011777202      -0.13477720
## E             14.44804        0.1400785      0.098318707      -0.14058661
##   accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
## A      -50.3270609         52.66577        -55.91428         -384.9707
## B        0.2961825         70.13559        -15.75998         -251.9416
## C      -42.5511198         33.03116        -56.16991         -370.6397
## D      -24.0518135         54.03938        -36.02435         -323.3964
## E      -18.8618938         55.43095        -24.95751         -292.5640
##   magnet_dumbbell_y magnet_dumbbell_z roll_forearm pitch_forearm
## A          217.5875          11.31201     25.12806     -6.836275
## B          275.0132          49.26283     32.80668     15.153611
## C          159.3520          62.87244     59.78918     12.067931
## D          221.4047          57.64870     16.16031     28.028964
## E          237.5469          71.73303     39.72281     16.487760
##   yaw_forearm total_accel_forearm gyros_forearm_x gyros_forearm_y
## A   24.841924            32.30585       0.1765681      0.16659200
## B   10.290048            35.15050       0.1286880      0.12526986
## C   38.914046            35.14362       0.2118987      0.07508763
## D    5.958482            36.14611       0.1264093     -0.03365285
## E   12.411002            36.78383       0.1464804      0.04703464
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## A       0.1931750       -3.058841        168.5774       -57.89904
## B       0.1774419      -78.480913        130.1755       -45.20228
## C       0.1508082      -47.727361        214.6251       -60.27361
## D       0.1073731     -155.313472        155.3254       -48.32642
## E       0.1445312      -70.681755        148.2079       -59.67991
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z
## A        -200.7593         470.7553         406.4719
## B        -323.7907         267.2343         369.4656
## C        -336.2405         506.5482         471.0959
## D        -461.1031         323.4383         362.5197
## E        -331.8859         281.7427         355.6993
## 
## Coefficients of linear discriminants:
##                                LD1           LD2           LD3
## roll_belt             5.814206e-02  9.461804e-02 -1.178255e-02
## pitch_belt            3.241396e-02 -4.368517e-03 -6.802217e-02
## yaw_belt             -9.328895e-03 -2.575229e-04 -1.224681e-02
## total_accel_belt     -1.962444e-02 -6.831768e-02 -2.686766e-01
## gyros_belt_x          5.759884e-01  1.169077e-01  9.733409e-01
## gyros_belt_y         -1.608609e+00 -2.236457e+00 -6.575377e-01
## gyros_belt_z          6.406957e-01  6.063942e-01  3.923319e-01
## accel_belt_x         -4.372073e-03 -1.100864e-03  2.616620e-02
## accel_belt_y         -2.437203e-02 -2.696288e-02  6.565370e-02
## accel_belt_z          4.590583e-03  2.533173e-02 -1.691194e-02
## magnet_belt_x        -1.094344e-02 -2.522016e-05 -2.422252e-02
## magnet_belt_y        -2.133530e-02 -9.264278e-03  2.600943e-03
## magnet_belt_z         7.532924e-03  1.450398e-03  1.169414e-02
## roll_arm              8.984905e-04  5.667449e-04  2.598171e-03
## pitch_arm            -2.477055e-03  6.872185e-03  5.372807e-03
## yaw_arm               1.458603e-03 -6.243406e-04  2.080235e-03
## total_accel_arm       5.421381e-03 -2.388513e-02 -2.104302e-02
## gyros_arm_x           1.286746e-01 -1.294543e-03 -5.854960e-02
## gyros_arm_y           8.268359e-02 -8.600223e-02 -8.894515e-02
## gyros_arm_z          -1.806505e-01 -1.222656e-01 -1.292026e-02
## accel_arm_x          -3.220171e-03 -5.659673e-03 -7.100260e-03
## accel_arm_y          -2.585356e-03  1.469207e-02 -1.510855e-03
## accel_arm_z           1.003179e-02 -1.789596e-03  1.611519e-03
## magnet_arm_x          4.677265e-05 -8.858255e-05  2.012036e-03
## magnet_arm_y         -1.700268e-03 -4.471301e-03  5.364170e-03
## magnet_arm_z         -3.814392e-03 -2.843368e-03 -5.089795e-03
## roll_dumbbell         2.492114e-03 -3.861204e-03 -2.689326e-03
## pitch_dumbbell       -5.600601e-03 -3.402566e-03 -4.485494e-03
## yaw_dumbbell         -7.364979e-03  6.599230e-03 -4.389308e-03
## total_accel_dumbbell  7.241640e-02  6.258644e-02  2.148833e-03
## gyros_dumbbell_x      2.406461e-01 -4.840172e-01  3.277621e-01
## gyros_dumbbell_y      2.121910e-01 -3.203740e-01  9.844321e-02
## gyros_dumbbell_z      9.110781e-02 -2.767363e-01  2.200709e-01
## accel_dumbbell_x      1.307495e-02  8.539629e-03  2.064877e-03
## accel_dumbbell_y      2.406728e-03  3.071405e-03  4.945168e-03
## accel_dumbbell_z      2.427409e-03  2.213680e-03  2.326650e-03
## magnet_dumbbell_x    -4.428691e-03  3.063147e-04  2.414387e-03
## magnet_dumbbell_y    -9.695553e-04  2.218822e-03 -1.264788e-03
## magnet_dumbbell_z     1.318981e-02 -1.023898e-02  4.655998e-04
## roll_forearm          1.617487e-03  1.267429e-03  2.345274e-04
## pitch_forearm         1.554783e-02 -1.228649e-02  6.620302e-03
## yaw_forearm          -2.127039e-04  8.797312e-04  1.888760e-04
## total_accel_forearm   3.093460e-02  4.534956e-03 -7.379033e-03
## gyros_forearm_x      -4.265090e-02 -1.964678e-02  1.614998e-01
## gyros_forearm_y      -1.044709e-02 -1.730787e-02  6.718517e-03
## gyros_forearm_z       4.027779e-02  6.843398e-02 -2.893955e-02
## accel_forearm_x       3.478111e-03  1.058354e-02 -1.419974e-03
## accel_forearm_y       9.753543e-04 -1.241123e-03 -6.668216e-04
## accel_forearm_z      -7.286756e-03  3.448147e-03  3.661783e-03
## magnet_forearm_x     -1.823715e-03 -3.415448e-03  4.660017e-04
## magnet_forearm_y     -9.706362e-04 -1.350985e-03  6.031495e-04
## magnet_forearm_z     -1.967850e-04 -1.471127e-03 -8.606643e-05
##                                LD4
## roll_belt             0.0692809787
## pitch_belt            0.0083022179
## yaw_belt             -0.0044542616
## total_accel_belt     -0.1675910452
## gyros_belt_x          0.3157486263
## gyros_belt_y          0.7640729669
## gyros_belt_z         -0.5965644356
## accel_belt_x          0.0087845529
## accel_belt_y          0.0096642905
## accel_belt_z          0.0149707036
## magnet_belt_x        -0.0044259309
## magnet_belt_y        -0.0025594635
## magnet_belt_z         0.0039712345
## roll_arm              0.0007978089
## pitch_arm             0.0014996887
## yaw_arm              -0.0013662757
## total_accel_arm      -0.0220440850
## gyros_arm_x           0.0139237565
## gyros_arm_y           0.1099127045
## gyros_arm_z           0.0889899706
## accel_arm_x          -0.0028429081
## accel_arm_y           0.0038683777
## accel_arm_z          -0.0080412930
## magnet_arm_x          0.0014234978
## magnet_arm_y          0.0008547717
## magnet_arm_z          0.0018798460
## roll_dumbbell        -0.0081003489
## pitch_dumbbell       -0.0036129595
## yaw_dumbbell         -0.0034794989
## total_accel_dumbbell  0.0051325802
## gyros_dumbbell_x      0.0467446715
## gyros_dumbbell_y      0.2103414907
## gyros_dumbbell_z      0.0196324185
## accel_dumbbell_x      0.0055355696
## accel_dumbbell_y     -0.0023273291
## accel_dumbbell_z      0.0015351394
## magnet_dumbbell_x    -0.0021338190
## magnet_dumbbell_y    -0.0019936379
## magnet_dumbbell_z     0.0093832502
## roll_forearm          0.0011837238
## pitch_forearm         0.0011990994
## yaw_forearm           0.0011007232
## total_accel_forearm   0.0064160934
## gyros_forearm_x       0.1466249332
## gyros_forearm_y      -0.0008258897
## gyros_forearm_z       0.0018593985
## accel_forearm_x       0.0041712248
## accel_forearm_y      -0.0021623655
## accel_forearm_z      -0.0041692217
## magnet_forearm_x     -0.0012449781
## magnet_forearm_y      0.0003070619
## magnet_forearm_z      0.0011873977
## 
## Proportion of trace:
##    LD1    LD2    LD3    LD4 
## 0.4888 0.2450 0.1532 0.1130
```

```r
## Find out of sample error rate

confusionMatrix(predict(modelFitLDA,newdata = testing),testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1805  225  134   91   58
##          B   43  969  120   54  241
##          C  193  191  888  160  122
##          D  181   52  184  931  145
##          E   10   81   42   50  876
## 
## Overall Statistics
##                                           
##                Accuracy : 0.697           
##                  95% CI : (0.6867, 0.7072)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6167          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8087   0.6383   0.6491   0.7240   0.6075
## Specificity            0.9095   0.9276   0.8972   0.9143   0.9714
## Pos Pred Value         0.7804   0.6790   0.5714   0.6236   0.8272
## Neg Pred Value         0.9228   0.9145   0.9237   0.9441   0.9166
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2301   0.1235   0.1132   0.1187   0.1116
## Detection Prevalence   0.2948   0.1819   0.1981   0.1903   0.1350
## Balanced Accuracy      0.8591   0.7830   0.7732   0.8191   0.7895
```

This gives an accuracy of 69.7% which is better than regression classification but not as good of Random Forest.


Conclusion:
-----------------------------

Best prediction model is random forest which gives an accuracy of 95.09% . 


Test Set Prediction
--------------------------

We will use random forest model with PCA for predicting "classe" variable for test set.


```r
testingDataPC<- predict(preProc,testingData[,-160])

predictions<-predict(modelFitRF,newdata = testingDataPC)

predictions
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

References:
-------------------------------------------
WLE Dataset:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4VgcBgFsr
