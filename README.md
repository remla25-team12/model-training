# Model Training
![pylint](https://img.shields.io/badge/PyLint-6.11-orange?logo=python&logoColor=white)

This repository contains the machine learning pipeline for training the sentiment analysis model used in our application.

## Repository Structure

- `src/`
  - `train_model.py`: Main script to run the training pipeline.
  - `get_data.py`: Script to download the dataset from the source repository.
- `model/`
  - `Classifier_Sentiment_Model.joblib`: Trained sentiment classification model (Naive Bayes).
  - `c1_BoW_Sentiment_Model.pkl`: Fitted Bag-of-Words vectorizer used during training.

## Preprocessing Logic

All preprocessing logic is factored out into the [`lib-ml`](https://github.com/remla25-team12/lib-ml) library.

## How to Train the Model

1. Clone the repository:
   ```bash
   git clone https://github.com/remla25-team12/model-training
   cd model-training
   ```
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Run the training pipeline:
   ```bash
   python3 src/train_model.py
   ```
   
## Output

After training, the following models are saved to `model/`:

- `Classifier_Sentiment_Model.joblib`: Trained sentiment classification model.
- `c1_BoW_Sentiment_Model.pkl`: Fitted Bag-of-Words vectorizer.
