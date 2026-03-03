# Football-Prediction
Using EPL data from 2000-2016


Football Match Outcome Prediction using Machine Learning
📌 Project Overview
This project aims to predict the outcome of English Premier League (EPL) matches using Machine Learning. Specifically, it performs binary classification to determine whether the Home Team (1) or Away Team (0) wins a match.
The model is trained on historical EPL match data across multiple seasons, with a time-based split to ensure realistic evaluation and prevent data leakage.


🎯 Problem Statement
Predict match outcomes using historical football statistics while avoiding future data leakage.
Training Data: All seasons before 2017
Testing Data: 2017 & 2018 seasons
Target Variable: FTR
1 → Home Win
0 → Away Win
📊 Dataset Description
The dataset contains structured football match statistics grouped into the following categories:
1️⃣ Match Information
Date
HomeTeam
AwayTeam
2️⃣ Season Statistics
HTGS (Home Team Goals Scored)
ATGS (Away Team Goals Scored)
HTGC (Home Team Goals Conceded)
ATGC (Away Team Goals Conceded)
DiffPts
3️⃣ Form & Momentum Features
Recent match performance points
Win/Loss streak indicators
🧹 Data Cleaning & Preprocessing
To ensure model integrity and avoid leakage:
Removed post-match columns (FTHG, FTAG)
Dropped duplicate and redundant features
Eliminated low-relevance columns (e.g., Matchweek)
Removed highly correlated features using correlation heatmaps
Applied StandardScaler before model training


⭐ Feature Engineering
Custom Team Performance Score (0–100)
To convert categorical team names into meaningful numerical representations, a strength score was developed based on:
Win Rate (50%) – Overall consistency
Goal Efficiency (30%) – Goals scored vs conceded
Streak Stability (20%) – Momentum factor
✔ Calculated using only training data to prevent leakage
✔ Captures overall historical team performance in a single metric
📈 Exploratory Data Analysis (EDA)
Key insights from analysis:
Correlation heatmaps (before & after cleaning) improved feature independence
Boxplots confirmed natural team strength differences
Goal histograms showed balanced scoring distributions
Team vs Year heatmap revealed long-term club performance trends
🤖 Models Implemented
Logistic Regression
Linear, fast, interpretable
Bagged Logistic Regression (100 models)
Reduces overfitting via averaging
Decision Tree
Simple but prone to overfitting
Gaussian Naive Bayes
Fast but assumes feature independence
Random Forest (100 Trees) ⭐ (Best Performing Model)
Ensemble method reducing overfitting
Handles correlated features effectively
Uses log-loss splitting criterion
Robust to noise and irrelevant features

🏆 Results
The Random Forest model achieved the best performance due to:
Ensemble learning approach
Reduced variance compared to single decision trees
Strong handling of correlated football statistics
Better probability-based classification
📌 Conclusion
This project demonstrates how proper data preprocessing, feature engineering, and time-based validation can significantly improve predictive performance in sports analytics. By combining statistical insights with ensemble learning techniques, the model effectively predicts football match outcomes while maintaining robustness and avoiding data leakage.
