# Customer Churn Model

Evaluate the causes of customer churn and create a machine learning algorithm to predict potential trends.

## Background
For our culminating project, we undertook a comprehensive analysis of customer churn data pertaining to an ecommerce firm. Our investigation encompassed an in-depth examination of various factors, including demographics, purchase behavior, and usage preferences. By evaluating these variables alongside additional pertinent factors, we sought to uncover their relationship with customer churn through the utilization of machine learning.

## Methodology 

First, we downloaded two datasets, one from Kaggle for the ecommerce company, and one from Open Intro for a randomized sample of individuals from census data. The data was uploaded into Amazon's AWS S3 platform for ease of importing and future-proofing the process. The S3 bucket was then opened in Google Colab and imported into a Pandas dataframe. The two datasets were merged based on randomizing a combination of the gender and marital status of the individual customer. The data was then cleaned and normalized, removing all rows containing null values and standardizing labeling across the dataset. The cleaned dataset was then uploaded back to S3.

The average personal income of the dataset was calculated and then matched against Census data to determine what city best resembled the sample. The city was determined to be Detroit, Michigan.

## Analysis 

Our initial analysis revealed a market penetration 0.98% with customer churn of 17.8%, using the Detroit, Michigan total population. The data was worked through matplotlib to evaluate the importance of different factors.

![image](https://github.com/rhisehl/Customer-Churn-Model/assets/116215793/87a024fb-f6d7-445f-ac92-456349cba98e)

The majority of the individuals in this sample have a tenure of less than 10 months, with an outlier at 50 months.


![image](https://github.com/rhisehl/Customer-Churn-Model/assets/116215793/56d2cc72-bd7c-47f8-96c0-38c7f267c457)

Most customers had personal incomes under $150,000 / year, but a few outliers had incomes up to almost $800,000. This is an expected outcome, with the largest portion of customers under $100,000 / year.


![image](https://github.com/rhisehl/Customer-Churn-Model/assets/116215793/4c07567c-18cd-4c30-ad0b-08a695361bdd)

The majority of the individuals in this sample were male (60%). Male and female churn were comparable, at 18.8% versus 16.3%, respectively.


![image](https://github.com/rhisehl/Customer-Churn-Model/assets/116215793/10e58f01-3ce1-4114-add1-b7904dca13b2)

71.4% of customers did not file a complaint in the preceding month, and those who did file a claim were approximately three times more likely to churn (33.2% vs 11.6%) 


![image](https://github.com/rhisehl/Customer-Churn-Model/assets/116215793/4dd80792-f0ce-454f-8b5a-2ec9cbdf26c2)

70.6% of customers preferred to use a mobile phone for interacting with the company. Those who used a computer were slightly more likely to churn (20.6%) compared to those who used a mobile phone (16.6%).


![image](https://github.com/rhisehl/Customer-Churn-Model/assets/116215793/6c36a393-5ee5-4185-b6d3-fa824dfa05d0)

Customer satisfaction scores show a trend when comparing churn, but it is unusual. An assumption was made that a 1 in satisfaction was poor and a 5 was good, however the trends are not reflective of this. For both churn and non-churn individuals, the highest probability was a score of 3. Those who did churn were much more likely to score between 2.5 and 5. Those who did not churn, however, were more likely to score a 1. Those who did not churn, in general, had a low chance of scoring half scores, while those who did churn had a more continuous plot. It is possible that the satisfaction score could be the inverseof expectations, more consultation with the company is needed to evaluate this trend.



## Machine Learning Model: Testing Phase
To begin the machine learning process, an inventory was taken of several models that could fit the data. These were:
* K-Nearest Neighbors
* Naive Bayes
* Logistic Regression
* Support Vector Machine
* Decision Tree
* Random Forests
* Neural Network

Next we analyzed the above models (except Neural Network) with and without Random Oversampling, and scaled the datasets using the following scalers:
* StandardScaler
* MinMaxScaler
* PowerTransformer(method='yeo-johnson')
* RobustScaler(quantile_range=(25, 75))
* MaxAbsScaler

The Neural Network model was analyzed for the lowest loss based on the following parameters:
* MaxAbsScaler
* without and with Random Oversampling
* nodes of 8, 16, 24
* dropout probabilites of 0, 0.2
* learning rates of 0.01, 0.005, 0.001
* batch sizes of 32, 64, 128
Due to the nature, robustness, the 1 hour 12 minutes of run time with this technique, multiple parameters and the small dataset modeled, we chose not to pursue this model for final analysis.

## Machine Learning Model: Best Fit
Most of the models had an average f1 score at or above 0.79, with Logistic Regression at 0.75 and Naive Bayes being the low of 0.67. The highest f1 score was the Service Vector Machine (SVM), with a 0.91 f1 score, and higher average recall at 0.93. This led to the decision to hone in on the Service Vector Machine model. This model was originally allowed to run with minimal parameters. After analyzing the individual parameters including RandomOversampling and the Scaler used to normalize the dataset, the individual average f1 score of 0.91 (0.908911) was highest for the RobustScaler (quantile range of 25, 75) with RandomOversampling. This became the starting point for optimization.

Initial Service Vector Machine Models

![SVM ros=False RobustScaler](https://github.com/rhisehl/Customer-Churn-Model/commit/d6110742d7cf71f6ebe0d3eb9ff716918209abe8#diff-8f25e43cff80c84d613225028e704cb1d73040583c895c09a9cf0a3e71401b4d)

![SVM ros=True RobustScaler](https://github.com/rhisehl/Customer-Churn-Model/commit/d6110742d7cf71f6ebe0d3eb9ff716918209abe8#diff-09303ee1428e92eae0c70b55d5caf979297f92b728c45e1816d13b0b9c35b7c6)

## Service Vector Machine Model : Optimization
# Kernel Trick
We began with a simple kernel trick to determine if the most major parameter of changing the kernel could be all that would be required. The low f1 score was 0.27 for the Polynomial kernel and high f1 score was 0.85 for the RBF Kernel, much less than the original parameters's f1 score of 0.91. We needed a more robust method of optimization so we dived into sklearn's GridSearchCv. 

# Sklearn GRidSearchCV
Sklearn's GridSearchCV boasts 13 available parameters to tune this model, we started with the 3 major parameters, selecting the following:
* kernels: rbf, sigmoid, and linear (dropped poly due to poor result of 0.27 with the kernel trick)
* gamma: 1, 0.1, 0.01, 0.001
* C: 0.1, 1, 10, 100
This resulted in 240 fits tested. The best fit resulted with the following parameters:
* kernels: rbf
* gamma: 1
* C: 1

![SVM after Optimization](https://github.com/rhisehl/Customer-Churn-Model/commit/d6110742d7cf71f6ebe0d3eb9ff716918209abe8#diff-3ac8cc03c98b7b26af60162913275cd7693f73b0f465f2fe3710f343ad7d2f45)

Due to GridSearchCV optimization we increased our best score of 0.908911 to an amazing 0.995007




## Technologies Utilized

Python, GoogleColab, Canva, AWS
### Python Libraries
* Pandas
* Numpy
* CPI
* boto3
* Matplotlib
* Tensorflow
* iPython
* OS
* Pydotplus
* Scikit-Learn
* IMB-Learn
* Spark
* RegEx

## Datasources

https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
https://www.openintro.org/data/index.php?data=census
https://www.census.gov/quickfacts/fact/table/detroitcitymichigan,MI/PST045222

## References

https://towardsdatascience.com/feature-scaling-effect-of-different-scikit-learn-scalers-deep-dive-8dec775d4946
https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3
https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
https://www.youtube.com/watch?v=i_LwzRVP7bg Machine Learning for Everybody from freeCodeCamp.org
https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167




# Project Planning Resources

Timeline: https://docs.google.com/spreadsheets/d/1mUIbTPY1Xd29zLsRSzpnpTErIFET0L7rTnL7LN7NJe0/edit#gid=1709744959

Proposal: https://docs.google.com/document/d/1Jg5Apw0ZbXjgVBtCeLI9XQ2aM47DyHTDMnrufyetgco/edit#heading=h.29hau7r7hy0i

Working Colab workbooks: https://drive.google.com/drive/folders/1vLQERtrbVcWDDE27lp25ukbEb7T4u0q1
