# Analysis-Of-Student-Outcome-in-MOOC

Prediction of Outcome in Massive Open Online Course by making use of 
various Machine Learning classification algorithms. 

# Introduction:
Massive Open Online Courses (MOOCs) have the potential to enable free 
university-level education on an enormous scale.Coursera, edX, and Udacity 
offer MOOCs though some are being offered directly by a college or 
university.​ Student interaction with learning activities, such as viewing of
video lectures, undertaking of quizzes, posting in discussion forums, and
interacting with the courseware is captured.Data captured from MOOCs
can provide valuable information for educators by analysing the patterns
present in the behaviour of learners.

# Objective:
A common pattern observed is large early dropout and low completion rate of a particular course. 
Early dropout predictions will provide a framework for developing mechanisms in MOOCs that provide individualized guidance and 
small-group support, 

Increasing retention rates. Thus, an effective way to predict the outcome of a MOOC is needed. 
 
# Scope:
To predict the outcome of user course based on behavioural and 
demographic patterns. 

# Implementation:
In this study an extensive data pre-processing and analysis 
techniques have been performed and exploratory analysis is 
performed.

The dataset used in this paper was obtained from
Harvard University
It is divided into these main parts: 
(i)Dataset -Harvard Dataset

(ii)Data Pre-Processing and Imputation

(iii)Exploratory Data Analysis

After the pre-processing and exploratory data analysis we apply
supervised learning models and perform the comparative study of
those models.
Results and Analysis of these algorithms is performed.

# Data Pre Processing and Imputations:

Downcasting of integer and float values. 
The objects are converted into categorical values. 
Almost by 82​ %​ reduction in memory usage is obtained and thus 
further preprocessing can be done. 
FInal Memory reduced from 381MB to 69MB​ . 
 
# Imputation Techniques: 
 Deleting rows. 
 Replace with mean/median/mode. 
 Assigning unique category. 
 Predicting the missing values using supervised machine learning 
algorithms. 
 Using algorithms which support missing values.

# Exploratory data analysis
The correlation heatmap was applied to measure the dependency
between the behavioral data and learners certification.So the
certification is highly positive correlation with ndaysact which
coefficient value 0.68 and with no of events and no of chapters its
coefficient value is 0.64 so it is moderately positive correlation.
The plot of correlation heatmap, which indicates a positive
relationship between three behavioral attributes and the target
variable.
So the attributes that are highly correlated can be used to reduce the
dimensionality of the dataset.
Only one of the two attributes can be taken into the final
classification model.

![Correlation matrix](https://github.com/omkar2811/Analysis-Of-Student-Outcome-in-MOOC/blob/master/Output/Correleance_1.png)

Applying Supervised ML Algorithms: 
We are going to use the supervised machine learning algorithms for 
classification problem 
1.Random Forest. 

2.Decision Tree.

3.Gaussian Naive Bayes. 

4.Gradient Boosting. 

5.Linear Discriminant Analysis.

6.Logistic Regression.

7.Ada Boosting.

8.Multi Layer Perceptron(Neural Network). 

# Results
Comparison of Algorithms :

![Compasrison 1](https://github.com/omkar2811/Analysis-Of-Student-Outcome-in-MOOC/blob/master/Output/Comparison1.png)

# Refrences
[1] A. D. Ho et al., “HarvardX and MITx: The First Year of Open
Online Courses, Fall 2012-Summer 2013,” SSRN Electron. J., no.
1, pp. 1–33, 2014.

[2] A. D. Ho et al., “HarvardX and MITxௗ: Two Years of Open
Online Courses Fall 2012-Summer 2014,” SSRN Electron. J., no.
10, pp. 1–37, 2015.

[3] Raghad AL-Shabandar,Abir Hussain,Andy Laws
Robert Keight, Janet Lunn
Naeem Radi " Machine Learning Approaches to Predict Learning
Outcomes in Massive Open Online Courses", 2017



