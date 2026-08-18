# HR Analytics — Job Change Prediction

## ITI AI Summer Internship Project

This project was developed as part of the **AI Summer Internship at the Information Technology Institute (ITI)**.

The goal of the project is to build a machine learning system that predicts whether a data scientist is likely to be looking for a job change based on different demographic, educational, professional, and employment-related features.

The project covers the complete machine learning workflow, starting from data exploration and preprocessing and ending with model evaluation and Streamlit deployment.

---

## Dataset

The project uses the **HR Analytics: Job Change of Data Scientists** dataset from Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists

The dataset presents a **challenging classification problem**, particularly because of the imbalance between the target classes and the characteristics of the available features.

Therefore, achieving extremely high accuracy is not necessarily expected or the primary objective of this project. Model performance was evaluated using multiple metrics rather than relying on accuracy alone.

---

## Project Objectives

- Explore and understand the HR dataset
- Perform data cleaning and preprocessing
- Handle missing values
- Encode categorical variables
- Address class imbalance
- Train multiple machine learning models
- Perform hyperparameter tuning
- Compare model performance
- Optimize the classification threshold
- Evaluate models using appropriate classification metrics
- Deploy the final solution using Streamlit

---

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The dataset was explored to understand:

- Feature distributions
- Missing values
- Categorical and numerical features
- Target variable distribution
- Relationships between variables
- Class imbalance

### 2. Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features where required
- Preparing the data for machine learning models

A preprocessing pipeline was used to ensure that the same transformations were consistently applied during training and prediction.

### 3. Handling Class Imbalance

Since the target classes are not perfectly balanced, class imbalance was taken into consideration during model development.

The evaluation therefore focuses on metrics such as:

- Precision
- Recall
- F1-score
- Average Precision / PR-AUC

rather than accuracy alone.

---

## Models

Several machine learning models were trained and compared as part of the project.

The models were evaluated using cross-validation and hyperparameter tuning to find suitable configurations for the dataset.

Hyperparameter optimization was performed using **RandomizedSearchCV**, with **Average Precision** used as the main optimization metric.

---

## Model Evaluation

Because this is a challenging and imbalanced dataset, **accuracy alone does not provide a complete picture of model performance**.

The project therefore considers:

| Metric | Purpose |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted positive cases were actually positive |
| Recall | How many actual positive cases were successfully identified |
| F1-score | Balance between Precision and Recall |
| PR-AUC / Average Precision | Performance across different classification thresholds |

The classification threshold was also tuned to improve the balance between precision and recall and obtain a more suitable F1-score.

---

## Streamlit Deployment

The trained machine learning solution was deployed using **Streamlit**, allowing users to interact with the model through a web interface.

**Live Demo:**  
https://iti-summerintern-project-vptaq6vkugqvh6bcaqeinq.streamlit.app/

---

## Project Structure

```text
ITI-Summer-Intern-Project/
│
├── iti.ipynb
├── README.md
├── requirements.txt
│
└── app/
    └── streamlit_app.py
---
