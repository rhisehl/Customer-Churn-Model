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
* Support Vector Machines
* Decision Tree
* Random Forests
* Neural Network


## Machine Learning Model: Best Fit

## Machine Learning Model: Optimization



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

# Project Planning Resources

Timeline: https://docs.google.com/spreadsheets/d/1mUIbTPY1Xd29zLsRSzpnpTErIFET0L7rTnL7LN7NJe0/edit#gid=1709744959

Proposal: https://docs.google.com/document/d/1Jg5Apw0ZbXjgVBtCeLI9XQ2aM47DyHTDMnrufyetgco/edit#heading=h.29hau7r7hy0i

Working Colab workbooks: https://drive.google.com/drive/folders/1vLQERtrbVcWDDE27lp25ukbEb7T4u0q1
