# Module 17:  Credit Risk Analysis 

# Project Overview 
## Purpose of Module 17
In this module, we explored machine learning algorithms in relation to data analysis. Through python and Scikit-learn technology, we compared the strengths and weaknesses of machine learning models in efforts to assess how well a model classifies and predicts data. We used python to build and evaluate our models and Scikit-learn to implement algorithms — such as a logistic regression, decision tree, random forest, or support vector machine— and  interpret the results. We were introduced to supervised learning algorithms and were challenged, given a scenario, to determine which technique is best for a dataset. We also used ensemble and resampling techniques to improve model performance. Through this model, we can successfully build and evaluate models to predict results. 

## Overview of Assignment 
For this assignment, we were tasked with using machine learning to predict credit risk and create a more reliable loan experience. Our goal was to create and evaluate models to identify a good loan candidate and loan default rates. Given that credit risk is an inherently unbalanced classification problem, we had to oversample our data—with RandomOverSampler and SMOTE) — undersample the data —with ClusterCentroids — and use a combinatorial approach — with SMOTEENN— to resample the data and predict credit risk. To reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier models are performed to predict credit risk. With each model, the performance is evaluated with the accuracy score, confusion matrix, and classification report. The best performing model would then be used to predict credit risk. 

## Resources 
LoanStats_2019Q1.csv 
Numpy version 1.20.3
Scipy version 1.7.3 
Scikit-learn version 1.0.2
Conda version 4.11.0
Python version 3.8.8
Jupyter version Notebook 6.3.0

### Websites 
[EasyEnsembleClassifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html)

[BalancedRandomForestClassifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)

# Results 
To make predictions on credit card risk, I utilized imbalanced-learn and Scikit-learn libraries to perform and evaluate multiple machine learning techniques. After reading in the data as a dataframe, the data was cleaned by removing null values and converting our testing data to numerical values with the .get_dummies() method. Our target/testing variable **_loan_status_** was created and the remaining columns were assigned as our features/training variables. We discovered that there are 68470 low risk counts and only 347 high risk counts; thus requiring techniques to train and evaluate unbalanced classes. Using models that attempt to balance and reduce bias, we resampled our dataset, trained a logistic regression classifier, calculated accuracy scores, generated a confusion matrix and classification report for each of the following algorithms: 

## RandomOverSampler 
We oversampled our split data with the **_RandomOverSampler _** algorithm. We resampled the data to get 51366 low risk counts and an equal 51366 high risk counts. We tiger trained a logistic regression model to the resampled data and predicted values from the split data. The image below summarizes this process: 

<img width="484" alt="randomover" src="https://user-images.githubusercontent.com/92558842/154846889-9818c457-2204-4b7b-aca0-4fcba93fd35e.png">

Using the logistic regression, we find that the accuracy to predict credit risk is 63.37%. We find from the confusion matrix, that the logistic regression was better at predicting high credit risk occurrences than low risk occurrences. Out of the 101 high risk occurrences, 67 were predicted to be high risk (recall is 66.34%), whereas out of the 17104 low risk occurrences, 10330 were predicted to be low risk (recall is 60.39%). However, we find from the imbalanced classification report that there is a perfect precision score for low risk predictions wherein every low risk result was predicted correctly. The f1 score further indicates that the model is better at predicting low risk occurrences. 

## SMOTE 
We oversampled our data using the **_SMOTE_** algorithm. The data was resampled to create 51366 low and high risk occurrences. A logistic regression model was then fit to the resampled data to predict credit risk. The image below shows the results of oversampling the data with SMOTE: 

<img width="541" alt="smote" src="https://user-images.githubusercontent.com/92558842/154846895-ff9c3cb2-685d-4647-8774-0ee822318b81.png">

Using the logistic regression model, we find an accuracy score of 66.06% for this algorithm. From the confusion matrix, we find that the logistic regression model was better at predicting low risk occurrences. Out of the 17104 low risk cases, 11761 were predicted to be low risk (recall 68.76%), whereas out of the 101 high risk occurrences, 64 were predicted high risk (recall 63.37%). Looking at the imbalanced classification report, we find a perfect precision score of 1 for low risk predictions where every low risk result was predicted correctly. We further find support that the logistic regression was better at predicting low risk occurrences in the f1 score of the model where low risk occurrences had an f1 of 0.81 and high risk cases had an f1 score of 0.02. 

## ClusterCentroids 
We under sampled our data using the **_ClusterCentroids_** algorithm. The data was resampled from this algorithm to find 246 high and low risk cases. A logistic model was trained from the resampled data to predict credit risk. From this analysis, we find the following metric evaluation for the model: 

<img width="494" alt="undersample" src="https://user-images.githubusercontent.com/92558842/154846923-39880d34-56d8-4a92-9e2e-72909ae9cb9b.png">

Based on the accuracy score, we find that for this model, 54.42% of the cases were predicted correctly. From the confusion matrix, we find that the logistic regression model was better at predicting high risk cases. Out of the 101 high risk recordings, 70 were predicted to be high risk (recall 69.31%), whereas out of the 17104 low risk recordings, 6763 were predicted low risk (recall 39.54%). The imbalanced classification report on the other hand shows that the model has a perfect precision score for low risk cases, meaning that all of the low risk occurrences were predicted correctly. The f1 score of 0.57 for low risk cases is also better than the 0.01 f1 score of high risk cases, indicating that this model is better at predicting low risk cases. 

## SMOTEENN
We tested a combination of over and under sampling with the **_SMOTEENN_** algorithm. The data was resampled with SMOTEENN to train the data to have 68460 high risk cases and 62011 low risk cases. With the resampled data, we trained the logistic regression model to predict credit risk. The SMOTEENN algorithm was then evaluated to find the following metric evaluations: 

<img width="482" alt="smoteenn" src="https://user-images.githubusercontent.com/92558842/154846934-c7a55461-078a-4c15-9276-dab88a13a9b2.png">

Based on the accuracy score of the SMOTEENN algorithm, we find that 65.13% of the cases were predicted correctly. From the confusion matrix, we find that the logistic regression model was better at predicting high risk cases. Out of the 101 high risk cases, 73 were predicted to be high risk (recall 72.28%). As for the 17104 low risk cases,  9919 were predicted to be low risk (recall 57.99%). The imbalanced classification report, on the other hand, indicates that the model has a perfect precision score of 1 meaning every low risk result was predicted correctly and this is further reflected in a high f1 score of 0.73 for low risk cases when compared to the 0.02 f1 score for high risk cases. 

## BalancedRandomForestClassifier 
To improve the accuracy and decrease variance of the models at hand, we performed a **_BalancedRandomForestClassifier_** algorithm. The model was fit to the training data with 100 estimators to predict credit risk. The Balance Random Forest Classifier was then evaluated to find the following metrics: 

<img width="581" alt="balanced" src="https://user-images.githubusercontent.com/92558842/154846940-9dc2e5ad-a0a0-444d-9124-8393aa640d7d.png">

We find from the balanced accuracy score, that from the model, 76.35% of the cases were predicted correctly. We find from the confusion matrix, however, that the model is better at predicting low risk occurrences. Of the 17104 low risk cases, 14940 were correctly predicted to be low risk (recall 87.35%), whereas, if the 101 high risk cases, 66 were predicted to be high risk (recall 65.36%). These findings are further reflected in the imbalanced classification report where we find a perfect precision score of 1 and a nearly perfect f1 score of 0.93 for the low risk occurrences. 
With this model we are also able to rank the features of the data set by importances using the **.feature_importances_** method. Below is a sample of the dataframe containing each feature ranked from most important to least important where the top five listed are the most relevant to the dataset and the bottom five are the least relevant. 

<img width="143" alt="importancedf" src="https://user-images.githubusercontent.com/92558842/154846945-bcacb8cc-a147-42fa-bb41-e0a3f88a2f4e.png">


## EasyEnsembleClassifier 
To improve the accuracy of the models at hand, the balanced boosted learner, **_EasyEnsembleClassifier_** was applied to the dataset. The model was fit to the training data and values were predicted from the test data. The Easy Ensemble Classifier model was then evaluated to find the following metrics: 

<img width="555" alt="easyensemble" src="https://user-images.githubusercontent.com/92558842/154846958-7b2e9e04-ac11-4663-afcb-516238407c7e.png">

From the balanced accuracy score, we find that 93.29%of the cases were predicted correctly. From the confusion matrix, we find that the model is slightly better at predicting low risk occurrences. Of the 17104 low risk cases, 16164 were correctly predicted to be low risk (recall 94.50%). On the other hand, of the 101 high risk cases, 93 were predicted to be high risk (recall 92.08%). From the imbalanced classification report, we further find a perfect precision (score of 1) and nearly perfect (0.97) f1 score for low risk cases, indicating that the model is better at predicting low risk occurrences. 

# Summary 
In this assignment, we evaluated six algorithms to attempt to correctly predict credit risk from a given unbalanced dataset. In summary, we find the following results: 

- RandomOverSampler: 63.37% accuracy, 66.34% recall and 1% precision for high risk, 60.39% recall and 100% precision for low risk, and an f1 score of 0.02 for high risk and 0.75 for low risk 
- SMOTE: 66.06% accuracy, 63.37% recall and 1% precision for high risk, 68.76% recall and 100% precision for low risk, and an f1 score of 0.02 for high risk and 0.81 for low risk 
- ClusterCentroids: 54.42% accuracy, 69.31% recall and 1% precision for high risk, 39.54% recall and 100% precision for low risk, and an f1 score of 0.01 for high risk and 0.57 for low risk 
- SMOTEENN: 65.13% accuracy, 72.28% recall and 1% precision for high risk, 57.99% recall and 100% precision for low risk, and an f1 score of 0.02 for high risk and 0.73 for low risk 
- BalancedRandomForestClassifier: 76.34% accuracy, 65.35% recall and 3% precision for high risk, 87.35% recall and 100% precision for low risk, and an f1 score of 0.06 for high risk and 0.93 for low risk
- EasyEnsembleClassifier: 93.29% accuracy, 92.08% recall and 9% precision for high risk, 94.50% recall and 100% precision for low risk, and an f1 score of 0.16 for high risk and 0.97 for low risk 

To determine which model best predicts credit card risk, we must keep in mind the evaluated tradeoffs for binary classification and the challenge of determining the importance of precision versus recall for an imbalanced dataset like this. In both instances of mislabeling, the company providing the loans could potentially lose money; however, the instance of mislabeling a high credit risk as low (a false negative) is more harmful than mislabeling a low credit risk as high (a false positive). With the knowledge of false negatives being more harmful than false positives and the importance of fraud detection in this scenario, we would lean towards recall having more importance than precision; however a balance or a high f1 score is still important in binary classification. 
Based on the significance of recall, I would recommend the Easy Ensemble Classifier model where the recall for both high and low risk cases are above 90% and there is the highest f1 score record in comparison to the other algorithms. I would further recommend to scale the data and repeat the ensemble learners to see if the model evaluation metrics improve. 
