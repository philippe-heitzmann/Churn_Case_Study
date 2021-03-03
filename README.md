### Churn Case Study Overview

We tackle a case study on predicting churn at the single customer level for a Wealth Management business in order to produce an analytically-based strategy on how to deal with customer churn. Please refer to the Jupyter notebook for the Python code that went into this analysis. All other files in this repo provide code to launch a simple Dash and Flask app to create a dashboard that visualizes our results and provides real-time machine learning predictions and rankings of customers by expected value of prescribing promotional activity to stem predicted customer churn. 

### Analysis Summary

This case study analysis of customers churn rates in a Wealth Management business puts forward a decision function parameterized by predicted probability of churning for a customer and business costs associated with churning to rank customers by greatest expected value of prescribing or not prescribing promotional activity to stem attrition.

Based on the feature importances of our models, we recommend that Wealth Management advisors providing customers with investment advice look to reinforce relationships with customers that are (i) older, (ii) using a relatively higher number of products, (iii) are active members, (iv) have high balances and (v) identify as female. Assignment to the East USWM division being an important predictor in all our models may indicate (i) an emerging competitive threat in the East region that should be addressed or (ii) the need to investigate the East USWM division’s management practices in the case poor management is leading to increased customer attrition.

We finally create a dashboard experience using Dash and Flask Python web frameworks in order to make our customer ranking results available to Wealth Management advisors to better inform targeted marketing and sales outreach at the single-customer level.

##### Dashboard View:
![image](https://user-images.githubusercontent.com/8759492/109857750-025bb500-7c29-11eb-9280-586617c0a082.png)


