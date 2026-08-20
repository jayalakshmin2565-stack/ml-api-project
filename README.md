# ML API Project

## Project Overview

This project is a simple Machine Learning API for predicting the species of an Iris flower based on its flower measurements. The project will use the Iris dataset from scikit-learn and a Random Forest Classifier model. The trained model will later be served through a FastAPI REST API.

## Dataset

The project uses the built-in Iris dataset provided by scikit-learn.

The dataset contains four input features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The target is the Iris flower species.

## Machine Learning Problem

This is a classification problem.

The model will predict one of three Iris flower species:

* Setosa
* Versicolor
* Virginica

## Model

The project will use a Random Forest Classifier from scikit-learn.

The model will be trained using the Iris dataset and saved so that it can later be loaded by the FastAPI application.

## API Contract

The `/predict` endpoint will accept four flower measurements: sepal length, sepal width, petal length, and petal width. The API will validate the input values and pass them to the trained machine learning model. The model will predict the Iris flower species, and the API will return the predicted species in the response.

Example input:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example output:

```json
{
  "prediction": "setosa"
}
```

## Request to Response Flow

```text
User sends prediction request
        ↓
FastAPI receives the request
        ↓
Validate the input
        ↓
Send validated data to the ML model
        ↓
ML model predicts the Iris species
        ↓
FastAPI returns the prediction
```

## Project Goal

The goal of this project is to learn how to train, save, and serve a machine learning model through a REST API using FastAPI.

## Future Project Structure

```text
ml-api-project/
├── app/
│   ├── main.py
│   ├── models/
│   └── routers/
├── ml/
│   ├── train.py
│   └── saved_model/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```
