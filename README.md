
# Sentiment Analysis on Yelp Dataset

This repository contains a Jupyter Notebook for performing sentiment analysis on the Yelp dataset. The analysis leverages various natural language processing (NLP) techniques and machine learning models to predict the sentiment of Yelp reviews.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)

## Introduction

Sentiment analysis is a key aspect of understanding customer feedback and opinions. This project focuses on analyzing the sentiment of Yelp reviews to classify them as positive or negative. The notebook demonstrates the complete pipeline from data preprocessing to model evaluation.

## Dataset

The Yelp dataset contains reviews from the Yelp platform. In this notebook, the data is converted from a JSON format to CSV for easier manipulation and analysis. The dataset includes various attributes such as review text, user information, and ratings.

## Installation

To run the notebook, you'll need to install the following dependencies:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- NLTK
- pandas
- spacy
- textblob
- matplotlib

You can install these packages using pip:

```bash
pip install tensorflow nltk pandas spacy textblob matplotlib jupyter
```

## Usage

To use this notebook, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/sankadash/sentiment_analysis-yelp_restaurant_review-.git
    cd sentiment-analysis-yelp
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook:

    ```bash
    jupyter notebook Sentiment_Analysis_yelp.ipynb
    ```

4. Run the cells in the notebook to execute the sentiment analysis pipeline.

## Methodology

The notebook follows these key steps:

1. **Data Preprocessing**: Convert JSON to CSV, clean the text data by removing stop words, punctuation, and applying stemming/lemmatization.
2. **Feature Extraction**: Utilize techniques such as Bag of Words (BoW), TF-IDF, or word embeddings.
3. **Model Building**: Train a machine learning model (e.g., Logistic Regression, Naive Bayes, or a deep learning model using TensorFlow).
4. **Model Evaluation**: Evaluate the model performance using metrics like accuracy, precision, recall, and F1-score.
