# repurchase-prediction-classification

Repurchase prediction using binary classification models
The current project aims to analyse different patterns from customer and vehicle characteristics data. The goal is to predict if an existing customer is more likely to buy a new car using binary classification models.

## DATA DESCRIPTION

The dataset contains 17 different categorical features that include customer and vehicle characteristics. The records describe the unique customer and the car they bought in the company. The variable to be predicted is the target with label 1 if the customer has purchased more than 1 vehicle, 0 if they have only purchased 1.

## BUSINESS OBJECTIVES

The business project aims to deeply understand the data we encounter and get knowledge to perform binary classification models.
The results from the first stage can be used to create a pipeline of data cleaning and exploration adjusted to the data and its particularities. Stakeholders could use the final model results to predict if an existing customer is more likely to buy a new car or to continue new research that may include additional variables and enrich the current dataset.

## EXPERIMENT OBJECTIVES

The overall objective of this study is to fit several binary classification models using the described dataset to predict the given target variable and select the best model to perform a marketing campaign.
The project is divided into several experiments that allow further exploration to achieve a good enough model accuracy.
During the development of the data exploration and model fit, the expectations are to gain knowledge of the dataset, understand the behaviour of each variable and get insights for the further stages of the project.
The experiment also aims to produce a pipeline for data cleaning and exploratory data analysis (EDA) to select potential features that can explain the target.

## EDA

The EDA analysis gave valuable insights about the variable to understand if it has the potential to predict the behaviour of the target variable.

<img width="479" alt="image1" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/faa416dc-6af3-4305-b5ac-a09b71abfdc7">

<img width="459" alt="image2" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/3c357618-e9b1-4ef7-91e5-21361633457d">

<img width="414" alt="image3" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/9f1d50ef-2da2-4b52-8802-6c2c474ec51b">

<img width="596" alt="image4" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/8e9227c8-8550-4bfe-82ca-289e56059b8e">


## BUILDING THE FINAL MODEL

Several binary classification models were trained using the algorithms and based on the training data after data splitting.
For the final model, the study uses features based on an ordinal encoder for the age band, car segment, and car model variables. It also uses one hot encoding for gender variable.
Then, the study developed training models with a grid search for decision trees, random forests, and extra trees. Subsequently, the study experimented with different random searches with a random forest model looking to reduce overfitting.
The random forest analysis also includes a model with variables selected based on feature importance results. Moreover, a random forest model is also trained using one-hot encoding from the data cleaning process 2 to compare the performance of the two encoder approaches.

## BEST MODEL SELECTED
According to the score presented (94% train - 83% validation - 85% test), the model from experiment 5- 7.5 Random Forest Classifier with Random Search based on feature importance with 'max_depth': 25, 'max_features': 3, 'min_samples_leaf': 5, 'n_estimators': 76, was selected as the best model. 

<img width="572" alt="image5" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/24be6b55-a2fd-4f80-bdb2-dc2e5d334fb8">

## CONCLUSION
This study showed that using trees-based algorithms fit better the data. It is recommended to experiment with external variables to enrich the dataset.
It is recommended for the following stages of the study to experiment with additional features that might help to improve the learning process and reduce overfitting.


## REFERENCES

AGRAWAL, S. (2019). EDA for Categorical Variables—A Beginner’s Way. https://kaggle.com/code/nextbigwhat/eda-for-categorical-variables-a-beginner-s-way
Arsik36. (2020, August 7). Answer to ‘Scaling of categorical variable’. Stack Overflow. https://stackoverflow.com/a/63304313
Brownlee, J. (2020a, January 5). ROC Curves and Precision-Recall Curves for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
Brownlee, J. (2020b, January 7). Tour of Evaluation Metrics for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
Brownlee, J. (2020c, January 12). How to Fix k-Fold Cross-Validation for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
Brownlee, J. (2020d, January 26). Cost-Sensitive Logistic Regression for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/cost-sensitive-logistic-regression/
Brownlee, J. (2020e, May 24). Recursive Feature Elimination (RFE) for Feature Selection in Python. MachineLearningMastery.Com. https://machinelearningmastery.com/rfe-feature-selection-in-python/
Brownlee, J. (2020f, June 9). How to Use StandardScaler and MinMaxScaler Transforms in Python. MachineLearningMastery.Com. https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
Brownlee, J. (2020g, June 11). Ordinal and One-Hot Encodings for Categorical Data. MachineLearningMastery.Com. https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
Catbuilts. (2018, October 6). Answer to ‘difference between StratifiedKFold and StratifiedShuffleSplit in sklearn’. Stack Overflow. https://stackoverflow.com/a/52677641
Czakon, J. (2022, July 21). F1 Score vs ROC AUC vs Accuracy vs PR AUC: Which Evaluation Metric Should You Choose? Neptune.Ai. https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
DataCamp. (2020, January 1). Principal Component Analysis (PCA) in Python Tutorial. https://www.datacamp.com/tutorial/principal-component-analysis-in-python
DataCamp. (2023, February 1). K-Nearest Neighbors (KNN) Classification with scikit-learn. https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
David. (2019, January 4). Answer to ‘Training error in KNN classifier when K=1’. Cross Validated. https://stats.stackexchange.com/a/385572
Dodier, R. (2014, October 5). Answer to ‘Should I keep/remove identical training examples that represent different objects?’ Stack Overflow. https://stackoverflow.com/a/26199916
Ellis, C. (2021, August 24). Random forest overfitting. Crunching the Data. https://crunchingthedata.com/random-forest-overfitting/
Filho, M. (2023, March 24). Do Decision Trees Need Feature Scaling Or Normalization? https://forecastegy.com/posts/do-decision-trees-need-feature-scaling-or-normalization/
GeeksforGeeks. (2023, March 13). Principal Component Analysis with Python. GeeksforGeeks. https://www.geeksforgeeks.org/principal-component-analysis-with-python/
Hoffman, E. (2020, July 15). Tutorial: Exploratory Data Analysis (EDA) with Categorical Variables. Analytics Vidhya. https://medium.com/analytics-vidhya/tutorial-exploratory-data-analysis-eda-with-categorical-variables-6a569a3aea55
Hotz, N. (2018, September 10). What is CRISP DM? Data Science Process Alliance. https://www.datascience-pm.com/crisp-dm-2/
Kapkar. (2020). Which Machine Learning requires Feature Scaling(Standardization and Normalization)? And Which not? | Data Science and Machine Learning. https://www.kaggle.com/getting-started/a
Kumar, A. M. (2018, December 17). C and Gamma in SVM. Medium. https://medium.com/@myselfaman12345/c-and-gamma-in-svm-e6cee48626be
Kumar, A. (2020, July 27). MinMaxScaler vs StandardScaler—Python Examples. Data Analytics. https://vitalflux.com/minmaxscaler-standardscaler-python-examples/
Malato, G. (2021, June 7). Precision, recall, accuracy. How to choose? Your Data Teacher. https://www.yourdatateacher.com/2021/06/07/precision-recall-accuracy-how-to-choose/
Olugbenga, M. (2022, July 22). Balanced Accuracy: When Should You Use It? Neptune.Ai. https://neptune.ai/blog/balanced-accuracy
Pandas. (2023). pandas.get_dummies—Pandas 2.0.0 documentation. https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
Saxena, S. (2020, March 12). A Beginner’s Guide to Random Forest Hyperparameter Tuning. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
scikit-learn developers. (2023a). Sklearn.linear_model.LogisticRegression. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
scikit-learn developers. (2023b). Sklearn.metrics.fbeta_score. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.metrics.fbeta_score.html
scikit-learn developers. (2023c). 3.3. Metrics and scoring: Quantifying the quality of predictions. Scikit-Learn. https://scikit-learn/stable/modules/model_evaluation.html
scikit-learn developers. (2023d). Sklearn.model_selection.cross_val_score. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.model_selection.cross_val_score.html
scikit-learn developers. (2023e). Sklearn.neighbors.KNeighborsClassifier. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
14
scikit-learn developers. (2023f). Sklearn.svm.SVC. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.svm.SVC.html
scikit-learn developers. (2023g). Sklearn.ensemble.ExtraTreesClassifier. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
scikit-learn developers. (2023h). Sklearn.tree.DecisionTreeClassifier. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
scikit-learn developers. (2023i). Sklearn.metrics.make_scorer. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.metrics.make_scorer.html
scikit-learn developers. (2023j). Sklearn.model_selection.GridSearchCV. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
scikit-learn developers. (2023k). Sklearn.model_selection.StratifiedKFold. Scikit-Learn. https://scikit-learn/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
Sethi, A. (2020, March 5). One-Hot Encoding vs. Label Encoding using Scikit-Learn. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/
So, A. (2023). Course Modules: 36106 Machine Learning Algorithms and Applications—Autumn 2023. Lab Solutions. https://canvas.uts.edu.au/courses/26202/modules
