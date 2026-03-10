Crime Prediction and Police Deployment Recommendation System
A machine learning-based solution to predict crime counts in high-risk areas and generate data-driven police deployment recommendations using Random Forest Regression.
Overview
This project leverages historical crime data (Chicago 2001-present) to:
Predict crime occurrences at the block-hour level
Identify key factors influencing crime rates
Generate actionable police deployment recommendations for high-risk areas
Visualize crime patterns and deployment needs
Key Features
Data Preprocessing: Handles missing values, feature engineering, and categorical encoding
Predictive Modeling: Random Forest Regression for crime count prediction
Feature Importance Analysis: Identifies critical crime-influencing factors
High-Risk Area Detection: Focuses on top 30 high-crime blocks (24-hour prediction)
Police Deployment Logic: Evidence-based officer allocation (1 officer per 2 predicted crimes)
Visualization: Hourly crime distribution + high-risk block police requirements
Technical Stack
Programming Language: Python 3.x
Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn (RandomForestRegressor, LabelEncoder, train_test_split)
Visualization: Matplotlib