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

## Feature Scaling
To scale the feature Standard Scaler is used. Some algorithm needs feature scaling and some doesn't. If the selected machine learning model  required feature scaling then feature scaling will be done automatically else feature scaling will not performed.

### How standard scaler works?
Standardization rescales the feature such as mean is 0 and the standard deviation is 1. It's mean that it will scale the data and after scaling the mean of the data will be 0 and the standard deviation will be 1.

**Formula:** z=(x- μ)/σ

Here,

x=That value which you want to scale from a particular column

μ=mean of that column, from where you took X

σ=The standard deviation of that column, from where you took 
X First find the mean of data, then find the standard 
deviation. Then subtract mean from x and divide that with deviation.

## Model Training
In this project 4 different machine learning algorithms are used. The model which will perform well, will be selected among the four algorithm.

### **The algorithms are: **
#### **1. K-nearest Neighbors**
**How K-nearest Neighbors Regressor Works?**

KNN stands for k-nearest neighbor. The k-nearest neighbor method is used to predict the value of a new data point based on its k-nearest neighbor. In KNN regression if a new data point comes then it tries to find how many nearest neighbors are there around it. After getting the nearest neighbors it does mean of those selected neighbors value and the result is the new data point predicted value. Here k means that how many neighbors you want to compare. Suppose if you want to take 5 as k value, then a new data point will find the five nearest neighbors around it ,and will calculate those data points average. This average value will be the predicted value for the new data point. Always try to take odd values for k like 5,7,3 etc. To find which are the nearest data points it tries to find the distance between a new data point and other data points.

**To find the distance it uses,**

*two types method:*

*Euclidean*

Formula: ED=√{(X2-X1)2+(Y22-Y1)2}

*Manhattan*

Formula: MD= [ | x B - x A | + | y B - y A |]

Any of these methods(Euclidean or Manhattan) can be used.


![K-nearest Neighbors Regressor](https://github.com/Rafsun001/automlv3/blob/main/images/knn%20regressor.png?raw=true )

Suppose there are nine data points A=1,B=2,C=3,D=10,F=5,G=6,H=7,I=8,J=9. A new data points come Z.Now predict value of Z.
Let take the k value as three. Now Z data point will try to find the three nearest data points from it. The distance will be measured by the euclidean distance finding technique. After getting the three nearest data points, now it will do the mean of those three data points value. For example, here 3 nearest data points are C=3, B=2, H=7. So the mean is Z=(3+2+7)/3=4 So Z predicted value is 4.

**How K-nearest Neighbors Classifier Works?**

KNN stands for k-nearest neighbor. The k-nearest neighbor method is used to classify a new data point based on its k-nearest neighbor. In KNN classification if a new data point comes then it tries to find how many nearest neighbors are there around it. After getting the nearest neighbors its compares that how many neighbors are there from which class. New data points will be classified to that class, from which class most of the data points will come. Here k means that how many neighbors you want to compare. Suppose five data points are taken for k, then a new data point will find five nearest neighbors for classification. Always try to take odd values for k like 5,7,3 etc. To find which are the nearest data points it tries to find the distance between the new data point and other data points.

**To find the distance it uses,**

*two types method:*

*Euclidean*

Formula: ED=√{(X2-X1)2+(Y22-Y1)2}

*Manhattan*

Formula: MD= [ | x B - x A | + | y B - y A |]

Any of these methods(Euclidean or Manhattan) can be used.

![K-nearest Neighbors Classifier](https://github.com/Rafsun001/automlv3/blob/main/images/knn%20classification.png?raw=true)

Suppose there are 2 class X and Y. X class have four data points A, B, C, D, and class Y have five data points F, G, H, I, J. A new data points come Z. Now find where the Z should go or in which class Z should belong. At first take k value as three. Now Z data point will try to find the three nearest data points from it. The distance will be measured by the euclidean distance finding technique. After getting the three nearest data points, now it will see that from which class most of the data points was came. For example, here 2 nearest data points are from class X and one data point is from class Y. Because most of the data points are from class X so Z data points will be classified as class X.


#### **2. Support Vector Machine**
**Some keyword points to create SVR:**

*1. Kernel:* SVR performed regression at a higher dimension. Kernel function is used to map lower-dimensional data into higher dimensional data. It means that kernel is a function that maps lower-dimensional data to higher-dimensional data. There are three types of kernel used in SVR 1.Sigmoidal Kernel, 2.Gaussian Kernel,3.Polynomial Kernel etc.

*2. HyperPlane:* It is a line that helps to predict the target value.

*3. Boundary Line:* Two-lines are drawn around the hyperplane at a distance of epsilon(ε). We can say that all the data points which should be predicted lies inside these boundary line. Using the boundary line we create a tube where all the predicted data points lies.

*4. Support vectors:* Vector is used to create the boundary line. Here we select the some data points as support vectors and using those support vectors we use draw or create a boundary line.These data points lies close to the boundary according to the boundary line hyperplane will be created.

**How Support Vector Regressor Works?**

SVR(Support Vector Regressor) tries to fit the error in a certain threshold. In SVR what happens is that it tries to fit the data as much as possible in a given margin called epsilon(ε) tube. SVR tries to fit the error with respect to a threshold. It means you will define the boundary and the predicted value should not go outside of the boundary. In SVR we draw a straight line to predict the data is known as hyperplane.

![Support Vector Regressor](https://github.com/Rafsun001/automlv3/blob/main/images/svr.png?raw=true)

At first draw a line(hyperplane). Then draw two boundaries, on both sides of the hyperplane using support vectors. Support vectors are nothing but data points. Find the maximum distance between data points and the hyperplane. Those data points which distance is maximum select those data points as support vectors. Because here we are taking maximum distance of data points so there are no data points after these data points. Because support vectors are maximum distance data points and these data points are use as support vectors that's why all the data points lie inside of the boundary. Boundaries are basically a range or threshold that data can't go outside of the boundaries. SVR tries to fit as many as possible data inside of the boundary.
So all the data which you will predict are lies inside of these boundary lines and here hyperplane plays that role which best-fitted line plays in linear regression. To do the prediction hyperplane will be used.

**How Support Vector Classifier Works?**

The main idea of the support vector machine is that it finds the best splitting boundary between data. Here you have to deal with vector space, In this way, the separating line is a separating hyperplane. The maximum margin or space between classes is called the best-fitted line. The hyperplane is also called the decision boundary.

![Support Vector Classifier](https://github.com/Rafsun001/automlv3/blob/main/images/svc.png?raw=true)

Think that there are two different types of data. Now draw a line between these two different types of data which will separate all the data and the drawn line known as a hyperplane.
Now there is a question that how to draw the line because the line can be drawn from anywhere.
To draw the best line find the maximum distance between classes and then have to find a midpoint point in that distance and then draw a line on that midpoint. Two sides of the hyperplane draw two boundaries.

**How to draw boundaries?**

For this find, a data point from each class which is so close to each other class and these points are called support vector. After finding those points draw a line which will do just a light touch those points. Draw lines to the parallel of the hyperplane. Now you have hyperplane and boundaries. So boundary means the nearest data of each class from the hyperplane. Now if you sum the distance of two boundaries from the hyperplane then you will get the margin. Always choose that point for creating a hyperplane where you get maximum width or distance between classes because the maximum width of margin gives better accuracy.

#### **3. Random Forest **
** How Random Forest Regressor Works? **

Random forest is an ensemble bagging learning method. A decision tree is like structured algorithm that divides the whole dataset into branches, which again split into branches and finally get a leaf node that can't be divided. Random forests create multiple numbers of decision trees and these trees are called forests. This means, in decision tree we create one tree but in a random forest we create multiple trees and because there are too many trees so it is called a forest. To build a machine learning model the whole dataset divide into train and test. Random forest takes data points from the training dataset and data points are taken randomly that's why it is called random forest.

*Let's see an example:*

Suppose you got a job offer letter from a company. After getting the letter you should ask someone who works in that company or know many things about that company like, is that company is good or bad, how much salary they will give, office rule, working rules, etc. Now if you make a decision depending on one person's opinion that may be not right. Because that person can be frustrated or very satisfied with the company. But if you ask too many people and then do average/mean/median everyone opinion and then taking a decision can be more accurate. So here you can compare taking opinions from one person with a decision tree and taking a decision from multiple people with random forest. The facilities of doing this are now the prediction will be more accurate. In a random forest, every tree will give a predicted value and then you will do the average/mean of all predicted values given by each tree and the result of the average/mean will be the result means final prediction. You will get better Accuracy in random forests compared to the decision tree.

![Random Forest Regressor](https://github.com/Rafsun001/automlv3/blob/main/images/random%20forest%20regressor.png?raw=true)

Suppose there are 100 records in a dataset. Randomly take 25 records from that dataset and create a new small dataset or one bag, then again randomly take 25 data from the main dataset and create a 2nd bag or new dataset. Bags or new datasets will be created like this according to the needs.
Remember one thing is that, this selection is called replacement selection. It means that after creating one bag or one dataset from a random selection of data from the original dataset, the data you selected will go back in the main dataset again, and then again a new bag or new dataset will be created. It means bags can have duplicate records. This means the number one bag and number two bag can have the same records. Same records mean all the data can't be the same, some data can be the same like number 1 record of the main dataset is present in all the bags or some bags and it also can be possible that number 1 record is present in only one bag. After creating the bags or small datasets the number of decision trees will be used equally to the number of bags or datasets. If 5 bags or mini datasets are created then 5 decision tree models will be used. After completing the training each model will give an output. Then combine all those model outputs and then do average/mean/median. The result of average/mean/median is the final output or prediction.

** How Random Forest Classifier Works? **

Random forest is an ensemble bagging learning method. A decision tree is like a structured algorithm that divides the whole dataset into branches, which again split into branches and finally get a leaf node that can't be divided. Random forests creates multiple numbers of decision trees and these trees are called a forest. It means in decision tree we create one tree but in a random forest we create multiple trees and because there are too many trees so it is called a forest. To build a machine learning model the whole dataset is divided into train and test. Random forest takes data points from the training dataset and data points are taken randomly that's why it is called random forest.

*Let's see an example:*

Suppose you got a job offer letter from a company. After getting the letter you should ask someone who works in that company or know many things about that company like, is that company is good or bad, how much salary they will give, office rule, working rules, etc. Now if you make a decision depending on one person's opinion that may be not right. Because that person can be frustrated or very satisfied with the company. But if you ask too many people and then do vote everyone opinion and then taking a decision can be more accurate. So here you can compare taking opinions from one person with a decision tree and taking a decision from multiple people with random forest. The facilities of doing this are now the prediction will be more accurate. In a random forest, each tree will gives a predicted value and then you will do voting of all predicted values given by each tree and the result of the voting will be the output. You will get better Accuracy in the random forest to compare to decision tree.

![Random Forest Regressor](https://github.com/Rafsun001/automlv3/blob/main/images/random%20forest%20classifier.png?raw=true)


Suppose there are 100 records in a dataset. Randomly take 25 records from that dataset and create a new small dataset or one bag, then again randomly take 25 data from the main dataset and create a 2nd bag or new dataset. Bags or new datasets will be created like this according to needs. Remember one thing, this selection is called replacement selection. It means that after creating one bag or one dataset from a random selection of data from the original dataset, the data you selected will go back in the main dataset again, and then again a new bag or new dataset will be created. It means bags can have duplicate records. It means the number one bag and number two bag can have the same records. Same records means all the data can't be the same, some data can be the same like number 1 record of the main dataset is present in all the bags or some bags and it also can be possible that number 1 record is present in only one bag. After creating the bags or small datasets the number of decision trees will be used equally to the number of bags or datasets. If 5 bags or mini datasets are created then 5 decision tree models will be used. After completing the training each model will give an output. Then combine all those model outputs and then do vote. which output is mostly predicted is the final output.

#### **4. XGBoost** 
XGboost stands for extreme gradient boost. XGboost is an advanced technique of gradient boost. Decision trees are used in both boost and gradient boost models.

**Basic things which make XGboost advance from Gradient boost:**

**Lambda(λ):** It is nothing but a regularization parameter.

**Eta(η):** Eta is the learning rate. Learning rate means at what shift or speed you want changes the predicted value. In xgboost commonly eta value is taken as 0.3 but you can also take between 0.1-1.0

**Similarity score(SS):** SS=(sum of residuals)^2/number of residuals+λ.
When the sign of residuals is opposite then you will get a lower similarity score, it happens because the opposite sign similarity score cancels each other and if not then you will get a higher similarity score. Here lambda is used to control ss. Lambda adds a penalty in ss. This penalty helps to shrink extreme leaf or node weights which can stabilize the model at the cost of introducing bias.

**Gain** = S.S of the branch before split - S.S of the branch after the split

**Gamma(γ):** Gamma is a threshold that defines auto pruning of the tree and controls overfitting. If the gamma value is less than the gain value only then the split will happen in the decision tree otherwise the nodes will not split. So gamma decides that how far the tree will grow. The tree will grow until the value of gain is less than the gamma value. That moment when gain value more than gamma value, that moment growing of tree will stop.

Because XGBoost is an extreme or advanced version of gradient boost so the basic working process will be the same. So before learning about the xgboost working process you must know how gradient boost works. In XGBoost main working process like, find a basic prediction of the main target column by finding the mean of that target column, then find the residuals and then make residuals as the target, then train model, then again get new residuals as prediction, then add new residuals with basic prediction, then again find new residuals, then again train new model everything is same but the difference is how you create the tree, how you add tree predictions residual with basic prediction and what you will get by finding mean of the main target column.
To create a tree first take residuals for the root node. Then use conditions and split nodes like a normal tree. Trees are created in XGBoost normally like how you create in a decision tree. But here in each node, you calculate the similarity score. For one node and its child nodes, you calculate the gain value and you also have a value called gamma. With these values, you can control the overfitting of the decision tree and can get good accuracy. If the gamma value is less than the gain value then only the tree grows or that node can go forward otherwise not. By doing this you perform auto pruning and control overfitting.

**How to add ml prediction value with basic prediction:**
Formula: New prediction=previous prediction+(η*Model residual prediction)
So xgboost also works on residuals and tries to reduce it to get better prediction or accuracy like gradient boost. But here some extra parameters are used like gamma, eta, ss, gain, lambda or do some extra work to perform better from gradient boost.

*Let's see a example:*
This is the data and here dependent column is IQ.

| Age | IQ |
| ------ | ----------- |
| 20 | 38 |
| 15 | 34 |
| 10 | 20 |

Let's find the predicted value and residuals. To get the predicted value to calculate the mean of the dependent variable and here that is 30. To find the residuals subtract the dependent variable with the predicted value.

| Age | IQ | Predicted value | Residual |
| ------ | ------ | ------ | ------ |
| 20 | 38 | 30 | 8 |
| 15 | 34 | 30 | 4 |
| 10 | 20 | 30 | -10 |

*Lets calculate the similarity score:*

First put ƛ as 0

*Formula:* Similarity Score = (S.R2) / (N + ƛ)

Similarity Score(SS) = (-10+4+8)2 / 3+0 = 4/3 = 1.33

Now make a decision tree.

Let's set the tree splitting criteria for Age greater than 10(>10).

![XGBR](https://github.com/Rafsun001/laptop_recom_predic/blob/main/XGBR.png?raw=true )

Now calculate SS and the gain.

For left side leaves the ss:

SS=(-10)^2/ 1+0=100

For right side leaves the ss:

SS=(4+8)^2/ 2+0=72

gain=(100+72)-1.33

Now if the gain is greater than the gamma value only then the splitting will happen of these leaves. For example, let's take the gamma value as 135. So here gain value is greater than gamma so the splitting will happen.

**Now let's see the prediction:**

*Formula:* New prediction=previous prediction+(η*Model residual prediction)

Now put the values in the formula and get the new prediction. Then all those processes will happen again and again until the residuals become zero.
