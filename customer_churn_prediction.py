Project Report
Customer Churn Prediction
GitHub URL
https://github.com/guptarupali/UCDPA_RupaliGupta

Abstract
Customer attrition is a case of concern for telecom companies. It is important to attract new customers and at same time, avoid any existing customer attrition/churns to grow their revenue generating base. Looking at churn, different reasons trigger customers to terminate their contracts, for example better price offers, more interesting packages, bad service experiences or change of customers’ personal situations.
Churn analytics provides valuable capabilities to predict customer churn and also define the underlying reasons that drive it. The churn metric is mostly shown as the percentage of customers that cancel a product or service within a given period (mostly months). If a Telecom company had 10 million customers on the 1st of January and received 500K contract terminations until the 31st of January the monthly churn for January would be 5%.


Introduction
The case of customer churn is critical for any company as this can have big impact of companies’ profit. In this project, I wrote machine learning models to predict churn on an individual customer basis and take counter measures such as discounts, special offers or other gratifications to keep their customers.
The key challenge for this project is to predict if an individual customer will churn or not. This project is based on customer churn analysis is a typical classification problem within the domain of supervised learning.
In this project, I have designed a basic machine learning pipeline based on a sample data set from Kaggle and compared the performance of different model types. The pipeline used for this example consists of 8 steps:
•	Step 1: Problem Definition
•	Step 2: Data Collection
•	Step 3: Exploratory Data Analysis (EDA)
•	Step 4: Feature Engineering
•	Step 5: Train/Test Split
•	Step 6: Model Evaluation Metrics Definition
•	Step 7: Model Selection, Training, Prediction and Assessment
•	Step 8: Hyperparameter Tuning/Model Improvement

Dataset
•	The data set for this classification problem is taken from Kaggle and stems from the IBM sample data set collection (https://www.kaggle.com/blastchar/telco-customer-churn).

Implementation Process
Step 1: Problem Definition
•	The key challenge is to predict if an individual customer will churn or not. To accomplish that, machine learning models are trained based on 80% of the sample data. The remaining 20% are used to apply the trained models and assess their predictive power with regards to “churn / not churn”. A side question will be, which features actually drive customer churn. That information can be used to identify customer “pain points” and resolve them by providing goodies to make customers stay.
•	To compare models and select the best for this task, the accuracy is measured. Based on other characteristics of the data, for example the balance between classes (number of “churners” vs. “non-churners” in data set) further metrics are considered if needed.
Step 2: Data Collection
The data set for this classification problem is taken from Kaggle and stems from the IBM sample data set collection (https://www.kaggle.com/blastchar/telco-customer-churn).
The use case pipeline build-up is started with imports of some basic libraries that are needed throughout the case. This includes Pandas and Numpy for data handling and processing as well as Matplotlib and Seaborn for visualization.
For this project the data set (.csv format) is downloaded to a local folder, read into the Pycharm and stored in a Pandas DataFrame.

Step 3: Exploratory Data Analysis

After data collection, several steps are carried out to explore the data. Goal of this step is to get an understanding of the data structure, conduct initial preprocessing, clean the data, identify patterns and inconsistencies in the data (i.e. skewness, outliers, missing values) and build and validate hypotheses.
Meaning of Features
By inspecting the columns and their unique values, a general understanding about the features can be build. The features can also be clustered into different categories:

Classification labels
Churn — Whether the customer churned or not (Yes or No)
Customer services booked
•	PhoneService — Whether the customer has a phone service (Yes, No)
•	MultipleLines — Whether the customer has multiple lines (Yes, No, No phone service)
•	InternetService — Customer’s internet service provider (DSL, Fiber optic, No)
•	OnlineSecurity — Whether the customer has online security (Yes, No, No internet service)
•	OnlineBackup — Whether the customer has online backup (Yes, No, No internet service)
•	DeviceProtection — Whether the customer has device protection (Yes, No, No internet service)
•	TechSupport — Whether the customer has tech support (Yes, No, No internet service)
•	StreamingTV — Whether the customer has streaming TV (Yes, No, No internet service)
•	StreamingMovies — Whether the customer has streaming movies (Yes, No, No internet service)


Customer account information
•	Tenure — Number of months the customer has stayed with the company
•	Contract — The contract term of the customer (Month-to-month, One year, Two year)
•	PaperlessBilling — Whether the customer has paperless billing (Yes, No)
•	PaymentMethod — The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
•	MonthlyCharges — The amount charged to the customer monthly
•	TotalCharges — The total amount charged to the customer
Customers demographic info

•	customerID — Customer ID
•	Gender — Whether the customer is a male or a female
•	SeniorCitizen — Whether the customer is a senior citizen or not (1, 0)
•	Partner — Whether the customer has a partner or not (Yes, No)
•	Dependents — Whether the customer has dependents or not (Yes, No)

Hypothesis Building
Looking at the features included in data and connecting them to their potential influence on customer churn, the following hypotheses can be made:
•	The longer the contract duration the less likely it is that the customer will churn as he/she is less frequently confronted with the termination/prolongation decision and potentially values contracts with reduced effort.
•	Customers are willing to cancel simple contracts with few associated product components quicker and more often than complexer product bundles — for bundles customers value the reduced administrative complexity. They might also be hesitant to cancel a contract, when they depend on the additional service components (e.g. security packages).
•	Customers with spouses and children might churn less to keep the services running for their family.
•	Tenure, contract duration terms and number of additional services are assumed to be among the most important drivers of churn.
•	More expensive contracts lead to increased churn as the chances to save money by changing providers might be higher.
•	Senior citizens tend to churn less due to the extended effort associated with terminating contracts.



Data Exploration
Plot insights:
•	Churning customers have much lower tenure with a median of ca. 10 months compared to a median of non-churners of ca. 38 months.
•	Churning customers have higher monthly charges with a median of ca. 80 USD and much lower interquartile range compared to that of non-churners (median of ca. 65 USD).
•	TotalCharges are the result of tenure and MonthlyCharges, which are more insightful on an individual basis.
•	Senior citizens churn rate is much higher than non-senior churn rate.
•	Churn rate for month-to-month contracts much higher that for other contract durations.
•	Moderately higher churn rate for customers without partners.
•	Much higher churn rate for customers without children.
•	Payment method electronic check shows much higher churn rate than other payment methods.
•	Customers with InternetService fiber optic as part of their contract have much higher churn rate.


