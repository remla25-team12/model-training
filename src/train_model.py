import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from get_data import download_data
from libml import preprocess_dataset

# Training pipeline is taken from the following repository:
# https://github.com/proksch/restaurant-sentiment

# Download and load the dataset
# TODO: Should we make the dataset reading dynamic using the glob library?
download_data()
dataset = pd.read_csv('data/a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)

# Preprocess the dataset
X, y, cv = preprocess_dataset(dataset)
#TODO: maybe move this to preprocess_data.py

# Save the CountVectorizer model for later use during inference
bow_path = 'model/c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))

# Divide dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Train and fit a Naive Bayes classifier 
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Exporting NB Classifier to later use in prediction
joblib.dump(classifier, 'model/Classifier_Sentiment_Model.joblib') 

# Evaluate model performance
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))