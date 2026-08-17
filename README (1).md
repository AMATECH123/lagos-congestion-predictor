# Lagos Traffic Congestion Predictor

A machine learning system that predicts traffic congestion levels (Low, Medium, or High) across major Lagos routes. It uses time, weather, road conditions, and calendar context, and it runs as a live web app.

Live demo. [lagos-congestion-predictor-bts7o3po6rnm3qkaw5un8b.streamlit.app](https://lagos-congestion-predictor-bts7o3po6rnm3qkaw5un8b.streamlit.app/)

Built by Muideen Abogunrin. [GitHub](https://github.com/AMATECH123) and [LinkedIn](https://linkedin.com/in/muideen-abogunrin-116ab2a9).

This project was built as part of the 3MTT AI/ML Fellowship, Airtel NextGen, Cohort 1.

## What it does

Given a route, time of day, day of week, and current road or weather conditions, the model predicts whether traffic will be Low, Medium, or High. It was trained on 25,000 rows of Lagos specific traffic data across 10 major routes, including Third Mainland Bridge, Lekki Epe Expressway, Apapa Oshodi, and Ikorodu Road.

The model reaches a macro F1 score of about 0.82 and an accuracy of about 85% on data it never saw during training.

## Why this is a real ML problem

The raw dataset also contained average speed, travel time, and a congestion score. These are direct measurements of congestion, not causes of it. Training on them would let the model read the answer off an adjacent column, which produces near perfect but useless accuracy in the real world. Those columns were left out on purpose. The model only sees information you would actually have ahead of time, such as time, weather, road status, event flags, and estimated vehicle count. That is what makes the prediction useful for planning.

## How it works

**Preprocessing**
Numeric features such as hour, lanes, route length, rain intensity, visibility, and vehicle count are standardized. Categorical features such as route and day of week are one hot encoded. Binary condition flags such as accident, roadwork, and checkpoint pass through unchanged.

**Class imbalance handling**
Real Lagos traffic is skewed. Roughly 60% of the data is Low congestion, 25% is Medium, and 14% is High. Training on this directly would produce a model that is good at predicting Low and bad at catching High, which is the case that actually matters. SMOTE oversampling is applied only to the training set to balance what the model learns from. The test set stays untouched so evaluation is honest.

**Model**
The model is a Random Forest, an ensemble of about 100 decision trees, each trained on a random subset of the data. At prediction time every tree votes and the majority class wins. Tree depth and leaf size are capped, using max_depth of 12 and min_samples_leaf of 5. This shrank the saved model from 237MB to 3.3MB with only a small accuracy cost, which made it possible to deploy on free hosting.

**Evaluation**
The model is scored using macro F1 instead of raw accuracy. Macro F1 scores each class equally regardless of how common it is in the data, so the model gets fair credit for catching High congestion cases even though they are rarer than Low ones.

Estimated vehicle count and hour of day are the strongest predictors. This matches real world intuition, since traffic volume and time of day are usually what drive congestion.

## Tech stack

| Layer | Tool |
|---|---|
| Data processing and modeling | Python, pandas, scikit learn, imbalanced learn |
| Model | Random Forest Classifier, 100 trees |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud |
| Model serialization | joblib, compressed |

## Project structure

```
├── app.py                        Streamlit web app
├── lagos_traffic_pipeline.py     Full training pipeline. Covers EDA, preprocessing, model comparison, tuning, and export
├── lagos_traffic_model.joblib    Trained, compressed model, 3.3MB
├── requirements.txt              Python dependencies
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retraining the model

The full pipeline in lagos_traffic_pipeline.py covers data cleaning, leakage checks, EDA, preprocessing, a comparison of three models (Logistic Regression, Random Forest, and Gradient Boosting), hyperparameter tuning through GridSearchCV, and export. Swap in a new CSV with the same schema and run it again to retrain.

## Possible extensions

A regression model on speed or travel time could give a continuous ETA prediction alongside this classifier. A Flask or FastAPI endpoint could serve predictions instantly and connect to an n8n workflow for automated alerts. A live weather or event API could auto fill inputs instead of requiring manual entry.
