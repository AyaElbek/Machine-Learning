# Bank Campaign Customer Classification

## Overview
This project aims to classify customers from a Portuguese bank campaign dataset to predict whether a customer would subscribe to a term deposit or not. The study utilizes various machine learning models to achieve this classification and discusses the methodology and results.

## Authors
- Elbekova Aidai
- Bethelhem Samson Gebreegziabhier
- Joshua Adu

### Supervisors
- Prof. Dr. André Hanelt
- Steven Görlich, M. Sc.

## Table of Contents
1. [Introduction](#introduction)
2. [Research Background](#research-background)
3. [Methodology](#methodology)
   - [Data Description](#data-description)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Description](#model-description)
   - [Data Optimization](#data-optimization)
4. [Results](#results)
   - [Baseline Model Results](#baseline-model-results)
   - [Selected Model Results](#selected-model-results)
   - [Cost of Misclassification](#cost-of-misclassification)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction
With the rapid evolution of technology and the emergence of new players in the financial market, traditional banks are undergoing significant transformations to keep up with ever-changing customer needs. This study explores how targeted marketing campaigns impact client acquisition by examining a Portuguese banking campaign dataset.

## Research Background
Targeted marketing campaigns have become increasingly prevalent in the banking industry. Machine learning models have emerged as powerful tools for developing more accurate and effective marketing campaigns.

## Methodology
The methodology involves several stages, including data pre-processing, baseline model assessment, model optimization, and validation.

### Data Description
The dataset consists of 41,188 observations with 20 input variables capturing both qualitative and quantitative attributes of the client profile, campaign, and socio-economic factors.

### Data Preprocessing
- **Finding Missing Values**: Detected and handled missing values.
- **Encoding Categorical Data**: Applied ordinal and one-hot encoding.
- **Feature Selection**: Utilized correlation metrics and variance inflation factor.
- **Splitting Dataset**: Divided into training and test sets.
- **Feature Scaling**: Applied Min-Max scaling.
- **Random Oversampling**: Addressed class imbalance.

### Model Description
The study employs various machine learning models:
- Logistic Regression
- Random Forest
- Support Vector Machine
- Neural Networks
- K-Nearest Neighbors
- AdaBoost
- Naive Bayes
- Decision Trees

### Data Optimization
- **Scoring Parameters**: Evaluated models using metrics like F1 score, Precision, Recall, and AUC-ROC.
- **Hyperparameter Tuning**: Used Grid Search for optimization.
- **K-fold Validation**: Implemented to avoid overfitting.

## Results
### Baseline Model Results
Evaluated different machine learning models based on various scoring parameters.

### Selected Model Results
Optimized models and evaluated their performance metrics. Random Forest emerged as the most promising model.

### Cost of Misclassification
Analyzed the cost of misclassification and its impact on model performance.

## Conclusion
The study highlights the importance of machine learning models in predicting customer behavior and optimizing marketing campaigns. The Random Forest model showed the best performance among all models evaluated.

## References
- List of references used in the study.

## How to Run the Code

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/bank-campaign-classification.git
    ```
2. **Set your working directory** to the location of the `bank_campaign_classification.R` file.
3. **Install the required libraries** in R:
    ```R
    install.packages(c("tidyverse", "caret", "e1071", "randomForest", "nnet"))
    ```
4. **Execute the `bank_campaign_classification.R` script** in R.

## Contact
For any questions or further information, please contact [your email].
# Bank Campaign Customer Classification

## Overview
This project aims to classify customers from a Portuguese bank campaign dataset to predict whether a customer would subscribe to a term deposit or not. The study utilizes various machine learning models to achieve this classification and discusses the methodology and results.

## Authors
- Elbekova Aidai
- Bethelhem Samson Gebreegziabhier
- Joshua Adu

### Supervisors
- Prof. Dr. André Hanelt
- Steven Görlich, M. Sc.

## Table of Contents
1. [Introduction](#introduction)
2. [Research Background](#research-background)
3. [Methodology](#methodology)
   - [Data Description](#data-description)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Description](#model-description)
   - [Data Optimization](#data-optimization)
4. [Results](#results)
   - [Baseline Model Results](#baseline-model-results)
   - [Selected Model Results](#selected-model-results)
   - [Cost of Misclassification](#cost-of-misclassification)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction
With the rapid evolution of technology and the emergence of new players in the financial market, traditional banks are undergoing significant transformations to keep up with ever-changing customer needs. This study explores how targeted marketing campaigns impact client acquisition by examining a Portuguese banking campaign dataset.

## Research Background
Targeted marketing campaigns have become increasingly prevalent in the banking industry. Machine learning models have emerged as powerful tools for developing more accurate and effective marketing campaigns.

## Methodology
The methodology involves several stages, including data pre-processing, baseline model assessment, model optimization, and validation.

### Data Description
The dataset consists of 41,188 observations with 20 input variables capturing both qualitative and quantitative attributes of the client profile, campaign, and socio-economic factors.

### Data Preprocessing
- **Finding Missing Values**: Detected and handled missing values.
- **Encoding Categorical Data**: Applied ordinal and one-hot encoding.
- **Feature Selection**: Utilized correlation metrics and variance inflation factor.
- **Splitting Dataset**: Divided into training and test sets.
- **Feature Scaling**: Applied Min-Max scaling.
- **Random Oversampling**: Addressed class imbalance.

### Model Description
The study employs various machine learning models:
- Logistic Regression
- Random Forest
- Support Vector Machine
- Neural Networks
- K-Nearest Neighbors
- AdaBoost
- Naive Bayes
- Decision Trees

### Data Optimization
- **Scoring Parameters**: Evaluated models using metrics like F1 score, Precision, Recall, and AUC-ROC.
- **Hyperparameter Tuning**: Used Grid Search for optimization.
- **K-fold Validation**: Implemented to avoid overfitting.

## Results
### Baseline Model Results
Evaluated different machine learning models based on various scoring parameters.

### Selected Model Results
Optimized models and evaluated their performance metrics. Random Forest emerged as the most promising model.

### Cost of Misclassification
Analyzed the cost of misclassification and its impact on model performance.

## Conclusion
The study highlights the importance of machine learning models in predicting customer behavior and optimizing marketing campaigns. The Random Forest model showed the best performance among all models evaluated.

## References
- List of references used in the study.

## How to Run the Code

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/bank-campaign-classification.git
    ```
2. **Set your working directory** to the location of the `bank_campaign_classification.R` file.
3. **Install the required libraries** in R:
    ```R
    install.packages(c("tidyverse", "caret", "e1071", "randomForest", "nnet"))
    ```
4. **Execute the `bank_campaign_classification.R` script** in R.

## Contact
For any questions or further information, please contact [your email].
