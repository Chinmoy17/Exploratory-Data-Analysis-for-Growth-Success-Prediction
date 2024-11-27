# Exploratory Data Analysis (EDA) for SME Growth Success Prediction

This repository contains an Exploratory Data Analysis (EDA) notebook that performs data cleaning, feature engineering, and basic machine learning model training on a dataset related to the growth success prediction of small and medium-sized enterprises (SMEs). The dataset includes information such as company status, funding rounds, city, state, and other relevant features.

In this notebook, we:

- Load and clean the dataset.
- Perform exploratory data analysis (EDA) to understand distributions, correlations, and other key insights.
- Handle missing values and outliers.
- Train machine learning models to predict the status of SMEs (acquired or closed).
- Evaluate model performance with classification metrics and confusion matrix visualizations.

The goal is to gain insights into the factors contributing to SME success and create a simple model for predicting whether an SME is likely to be acquired or closed.

## Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

## Dataset

The dataset used in this analysis is the "D7a_SME Growth Success Prediction" dataset, which contains information about various SMEs, such as:

- **id**: Unique identifier for the company.
- **status**: Whether the company is acquired (1) or closed (0).
- **city**: City where the company is located.
- **state_code**: State code where the company is located.
- **founded_at**: The date when the company was founded.
- **closed_at**: The date when the company was closed (if applicable).
- **first_funding_at**: The date of the company's first funding round.
- **last_funding_at**: The date of the company's last funding round.
- **avg_participants**: Average number of participants in funding rounds.

This dataset contains **923 rows** and **49 columns**.

## Data Cleaning and Preprocessing

In this notebook, the following preprocessing steps are applied to the dataset:

1. **Converting the 'status' column**: The 'status' column is mapped to binary values (1 for acquired, 0 for closed).
2. **Dropping unnecessary columns**: Columns such as 'Unnamed: 0', 'Unnamed: 6', 'name', 'labels', and 'object_id' are removed.
3. **Handling missing values**: Rows with missing values are dropped.
4. **Feature encoding**: Categorical columns such as 'city' and 'state_code' are label-encoded to numeric values.
5. **Date feature engineering**: New features are created from the date columns, such as 'lifespan_days' and 'time_to_first_funding_days'.
6. **Removing "c:" prefix**: The 'id' column is cleaned by removing the "c:" prefix and converting the column to numeric.
7. **Dropping non-numeric columns**: Non-numeric columns are dropped for model training purposes.

The cleaned dataset is then saved as `cleaned_dataset.csv`.
## Exploratory Data Analysis (EDA)

Various visualizations and statistical analyses are performed to explore the dataset:

- **Histograms**: To visualize the distribution of numerical variables.
- **Boxplots**: To identify outliers in features such as 'avg_participants', 'is_gamesvideo', and 'is_top500'.
- **Correlation Matrix**: To explore correlations between numerical features.
- **Missing Values and Duplicates**: Check for missing values and duplicate rows.
- **Summary Statistics**: The dataset’s basic statistics, including count, mean, and standard deviation, are printed.

Key insights can be drawn from these visualizations to understand the distribution and relationships of features in the dataset.
## Model Training

We train a machine learning model to predict the 'status' of SMEs. The following steps are involved:

1. **Feature Selection**: Features are selected for training the model, with 'status' being the target variable.
2. **Data Splitting**: The dataset is split into training and testing sets (80% training, 20% testing).
3. **Model**: We use a logistic regression model to predict whether an SME is acquired or closed.
4. **Model Evaluation**: Accuracy, classification report, and confusion matrix are generated to evaluate model performance.

The model achieves an accuracy of approximately **95.65%**, but the performance on predicting 'acquired' SMEs is limited due to the class imbalance.

## Results and Visualizations

The model's performance is evaluated using the following:

- **Accuracy Score**: 95.65%
- **Classification Report**: Precision, recall, and F1-score for both classes (acquired and closed).
- **Confusion Matrix**: A heatmap visualization of the confusion matrix to show the distribution of true and predicted labels.

These metrics help us understand the strengths and weaknesses of the model and suggest areas for improvement, such as addressing class imbalance.

### Confusion Matrix Heatmap:
## Confusion Matrix

The confusion matrix is as follows:

|               | Predicted: 0 (Closed) | Predicted: 1 (Acquired) |
|---------------|-----------------------|-------------------------|
| **Actual: 0 (Closed)**  | 44                    | 0                       |
| **Actual: 1 (Acquired)** | 2                     | 0                       |

- **True Negatives (TN)**: 44 (Predicted 0, Actual 0)
- **False Positives (FP)**: 0 (Predicted 1, Actual 0)
- **False Negatives (FN)**: 2 (Predicted 0, Actual 1)
- **True Positives (TP)**: 0 (Predicted 1, Actual 1)


## Conclusion

This notebook provides a comprehensive analysis of SME growth success prediction using an EDA approach and machine learning. While the model performs well in predicting the 'closed' status, there is room for improvement in predicting 'acquired' SMEs, particularly due to the class imbalance.

Future work could focus on:

- Addressing the class imbalance using techniques like oversampling, undersampling, or using more advanced models.
- Tuning hyperparameters to improve model performance.
- Incorporating additional features or external datasets for better predictions.

Feel free to modify the notebook and experiment with other machine learning algorithms to improve predictions!

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

