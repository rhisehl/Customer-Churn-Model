# Customer Churn Model

Evaluate the causes of customer churn and create a machine learning algorithm to predict potential trends.

## Background

For our culminating project, we undertook a comprehensive analysis of customer churn data pertaining to an ecommerce firm. Our investigation encompassed an in-depth examination of various factors, including demographics, purchase behavior, and usage preferences. By evaluating these variables alongside additional pertinent factors, we sought to uncover their relationship with customer churn through the utilization of machine learning.

## Methodology 

First, we downloaded two datasets, one from Kaggle for the ecommerce company, and one from Open Intro for a randomized sample of individuals from census data. The data was uploaded into Amazon's AWS S3 platform for ease of importing and future-proofing the process. The S3 bucket was then opened in Google Colab and imported into a Pandas dataframe. The two datasets were 

We then utilized AWS to upload and store the scrubbed dataset 

To visualize the key factors identified in the data, we employed  Matplotlib. This allowed us to easily visualize the relationships between various factors and customer churn. This gave us additional insights into what factors should be focused on when creating our models.

## Analysis 

Our initial analysis revealed a market penetration 0.98% with customer churn of 17.8%.
Based on the average personal income from our analysis we were able to extrapolate from census data that our market was Detroit. 



## Technologies Utilized

Python, GoogleColab, Canva, AWS

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
