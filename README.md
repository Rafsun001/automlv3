# AutoML Project
This is a automl project. The target is to 
automate the whole process (data cleaning, outliers handling, handling high cardinality, EDA, feature selection, model training, parameter tuning, model evaluation) of machine learning and achieve the best accuracy. After completing the process it will generate a python file where all the necessary code will be written to perform machine learning on the given dataset.

## Data Cleaning
1. Dropping columns.
2. Renaming columns name.
3. Handling date-time column.
4. Handling columns data type. 
5. Clean values of the each column.
6. Filling missing values.

## Methods to fill missing values
### **1. KNN imputation**
K-Nearest Neighbors (KNN) imputation is a technique used to fill missing values in a dataset based on the values of its neighboring data points. 

**Steps in KNN Imputation:**

*1. Identifying Missing Values:* First, you identify the missing values in your dataset.

*2. Selecting K:* Determine the number of nearest neighbors (K) to consider. This is a crucial parameter and affects the imputation result. The choice of K involves a trade-off between bias and variance - higher K values lead to smoother imputations but may introduce bias.

*3. Calculating Distance:* For each missing value, compute its distance to all other data points in the dataset. Common distance metrics used include Euclidean distance, Manhattan distance, etc.

*4. Finding Nearest Neighbors:* Select the K nearest neighbors to the data point with the missing value based on the computed distances. These neighbors are the data points with the most similar attributes/features.

*5. Imputation:* Once the nearest neighbors are identified, impute the missing value by taking an average (for numerical values) or mode (for categorical values) of the values from these neighbors. The missing value is replaced with this aggregated value.

*6. Repeat for all Missing Values:* Iterate through all missing values in the dataset and repeat the process to impute each missing value.

### **1. Simple imputation**
Simple imputation is a basic technique used in data analysis to handle missing values by filling them in with estimated or calculated values. The idea is to replace missing data points with some plausible values to maintain the structure and integrity of the dataset. 

**Steps in Simple imputation:**

*1. Identify Missing Values:* First, you need to identify the missing values within your dataset. These missing values can occur due to various reasons such as errors in data collection, equipment failure, or human error.

*2. Choose Imputation Method:* There are several simple imputation methods, each with its own approach:

*..Mean/Median Imputation:* Replace missing values with the mean or median of the observed values in that column. This method is suitable for numerical data and is less sensitive to outliers compared to the mean.

*..Mode Imputation:* For categorical data, you can replace missing values with the mode (most frequent value) of the observed values in that column.

*..Last Observation Carried Forward (LOCF):* In time series data, you might use the last observed value to impute missing values until the next observation is available.

*..Random Imputation:* Assign random values from the distribution of observed values to the missing data points.

*..Constant Imputation:* Replace missing values with a predefined constant value (e.g., replacing missing values in a numerical column with zero).

*3 .Apply Imputation:* Once you've chosen the imputation method that suits your data type and context, apply it to the missing values in your dataset.

### **Fill using mean**
In this method we will compute the mean of numerical column and fill all the missing values of that column using the computed mean value.

### **Fill using median**
In this method we will compute the median of numerical column and fill all the missing values of that column using the computed median value.

### **Fill using mode**
In this method we will find the mode value of categorical  column and fill all the missing values of that column using the mode value.

## Outliers Handling
1. To visualize the outliers boxplot and distplot are used.
2. After detection is done using IQR(Inter Quartile Range) method. 
3. After detecting the outliers some outliers has been removed and some replaced with other value.
 
**How IQR Works?**

The Interquartile Range (IQR) is a statistical measure used to identify outliers in a dataset. It's based on the spread of the middle portion of the data and is calculated as the difference between the third quartile (Q3) and the first quartile (Q1).

*The steps to use IQR to detect outliers are:*

*1. Calculate the IQR:*
Find the first quartile (Q1), which represents the 25th percentile of the data.
Find the third quartile (Q3), which represents the 75th percentile of the data.
Calculate the IQR as IQR = Q3 - Q1.

*2. Define Outliers:*
Any value that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier.

*3. Identify Outliers:*
Check each data point in your dataset against the defined boundaries:
Values below Q1 - 1.5 * IQR are considered outliers on the lower end.
Values above Q3 + 1.5 * IQR are considered outliers on the upper end.

## Graphs and Charts used for EDA
1. Bar chart
2. Pie Chart
3. Histogram
4. Boxplot
5. Distplot
6. Heatmap
7. Scatterplot
8. Pairplot

## Encoding
Encoding is done using label and one-hot encoding method.

**How label encoding works?**

Label encoding, encodes our categorical values in one column. Suppose we have 3 values in a column shirt, pant and watch. What label encoder will do, it will convert shirt into 0,pant into 1, and watch into 2. Here it will not create any new column it will encode in the same column. It means in that column where it will find good there it will use 0, similarly for very good and bad it will use 1 and 2.

**How one-hot encoding works?**

One-hot encoding is a technique used in machine learning to convert categorical data, represented as labels or integers, into a format that can be provided to machine learning algorithms. It's particularly useful when dealing with categorical variables that don't have a numerical relationship between categories.

Suppose you have a categorical feature, such as "Color," with three categories: Red, Green, and Blue

*Step 1:*
Initially, these categories might be encoded as integers, for example:

Red: 0

Green: 1

Blue: 2

*Step 2:*

Each category is represented as a binary vector.
For each category, a vector of length equal to the number of unique categories is created.
Each category's binary vector has all zeros except for a single 1 in the position corresponding to the category index.

For our example:

Red becomes [1, 0, 0]

Green becomes [0, 1, 0]

Blue becomes [0, 0, 1]

## Dropping Low Variance Features
Dropping low variance features are depends on user. If user want to do then the low variance features will be removed else this step will be skipped. To remove low variance features VarianceThreshold method will be used.

### How VarianceThreshold method works?
The VarianceThreshold method is a feature selection technique used in machine learning to remove low-variance features from a dataset. It operates on numerical features and identifies those features that have variance below a specified threshold. 

**Here's how it works:**

*1. Variance Calculation:* For each feature (column) in the dataset, the VarianceThreshold computes the variance. Variance measures how much values in a feature vary or spread out from the mean. Features with low variance indicate that most of their values are close to a constant.

*2. Threshold Application:* The method compares the computed variances against a predefined threshold. Features with variance below this threshold are considered to have low variance.

*3. Feature Selection:* The VarianceThreshold then identifies and removes the features that fall below the specified threshold. This process effectively eliminates features that show little variation across the dataset. The assumption here is that these features might not provide much discriminatory information for predictive modeling tasks because they don't change much across samples.

*4. Output:* The result of applying VarianceThreshold is a dataset with reduced dimensionality, containing only the features that have variance above the specified threshold.

## Handling Multicollinearity 
To handle multicollinearity Variance Inflation Factor(VIF) will be used. If user want to perform multicollinearity only then multicollinearity will be performed else this step will be skipped.

### How Variance Inflation Factor Works?
If two or multiple independent variables are highly correlated with each other in a regression problem, then it is called multicollinearity. It means, because the independent variables are highly correlated with each other then we can predict one independent variable by another independent variable. This correlation can be a positive correlation or a negative correlation. This problem occurs only in a regression problem.

**Example:**
Let's take linear regression model for example:
Suppose we have two independent variables X1 and X2 and dependent variable Y.
