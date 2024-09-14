# Social Media Rumour Detection

## Overview
This project aims to detect rumors in social media posts using machine learning techniques. It involves preprocessing the text data, vectorizing it, training various classification models, and evaluating their performance.

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- nltk
- gensim

## Language Used
Python

### Dataset

The dataset used for this project contains social media posts along with various attributes. Here's a basic overview of the data format:

- **File Format:** CSV
- **Columns:**
  - **ID:** Unique identifier for each social media post.
  - **Description:** Text content of the social media post.
  - **#Tweets:** Number of tweets by the user.
  - **Date_time_creation_account:** Date and time when the user's account was created.
  - **Language:** Language of the post.
  - **#Followers:** Number of followers of the user.
  - **#Friends:** Number of friends or accounts the user is following.
  - **Date&Time:** Date and time of the post.
  - **#Favorite:** Number of favorites or likes for the post.
  - **#Retweet:** Number of retweets for the post.
  - **Another Tweet Inside:** Indicates if there's another tweet inside this one.
  - **Source:** Source device or platform used to post the content.
  - **Tweet ID:** Unique identifier for the tweet.
  - **Retweet ID:** ID of the retweet, if applicable.
  - **Quote ID:** ID of the quoted tweet, if applicable.
  - **Reply ID:** ID of the reply tweet, if applicable.
  - **Frequency of tweet occurrences:** Frequency of occurrences for the tweet.
  - **State of Tweet:** Indicates the nature of the tweet (e.g., rumor, non-rumor).

You can use this dataset to train and evaluate machine learning models for rumor detection in social media posts.


## Preprocessing
### Description
The text data in the 'Description' column of the dataset is preprocessed before feeding it into the models. The preprocessing steps include:
- Converting text to lowercase
- Tokenizing text
- Removing punctuation and special characters
- Removing stopwords
- Lemmatizing tokens

### Implementation
The preprocessing is implemented using Python libraries such as NLTK and regular expressions. The preprocess_text function is applied to the 'Description' column of the DataFrame.

```python
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation and special characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.isalnum()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply preprocessing to the 'Description' column
df['Description'] = df['Description'].apply(preprocess_text)
```

## Vectorization
### Description
Text data is converted into numerical form using vectorization techniques. Three main vectorization methods are used:
- TF-IDF Vectorization: Converts text data into TF-IDF (Term Frequency-Inverse Document Frequency) vectors.
- Bag of Words (CountVectorizer): Converts text data into frequency-based vectors.
- Word2Vec: Converts text data into dense word embeddings using the Word2Vec algorithm.

### Implementation
- TF-IDF vectorization is performed using the TfidfVectorizer from scikit-learn.
- Bag of Words vectorization is performed using the CountVectorizer from scikit-learn.
- Word2Vec embedding is trained using the Word2Vec model from the Gensim library.

## Models
### Description
Several classification models are trained to classify social media posts as rumors or non-rumors. The following models are used:
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- Gradient Boosting

### Implementation
The models are trained using the vectorized data. The performance of each model is evaluated using accuracy and F1-score metrics.

## Model Training and Evaluation
### Description
The dataset is split into training and testing sets. Each model is trained on the training set and evaluated on the testing set.

### Implementation
- The train_test_split function from scikit-learn is used to split the dataset.
- Each model is trained and tested using the respective vectorized data.
- Accuracy and F1-score metrics are computed for each model.

## Hyperparameter Tuning
### Description
Hyperparameter tuning is performed to optimize the performance of the SVM model.

### Implementation
- GridSearchCV is used to search for the best combination of hyperparameters for the SVM model.
- The best-performing SVM model is selected based on cross-validated accuracy.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define hyperparameters grid for SVM
svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

# Initialize GridSearchCV for SVM
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)
best_svm_classifier = svm_grid_search.best_estimator_
best_svm_accuracy = svm_grid_search.best_score_

print("Best SVM Accuracy after Hyperparameter Tuning:", best_svm_accuracy)
```

## Visualization
### Description
The accuracy and F1-score of each model are visualized for comparison.

### Implementation
- Matplotlib is used to create bar charts showing the performance of each model. Separate visualizations are created for TF-IDF and Bag of Words vectorization methods.

## Result and Comparison
### Description
The performance of each model is compared based on accuracy and F1-score metrics.

### Result
- SVM Accuracy: 0.6498
- Naive Bayes Accuracy: 0.6179
- Random Forest Accuracy: 0.6341
- Gradient Boosting Accuracy: 0.5906

### Visualization

To visualize the accuracy of different classification models using TF-IDF vectorization, you can use the following code snippet:

```python
import matplotlib.pyplot as plt

# Plot model accuracies for TFIDF
models = ['SVM', 'Naive Bayes', 'Random Forest', 'Gradient Boosting']
accuracies = [svm_accuracy, nb_accuracy, rf_accuracy, gb_accuracy]
plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['#004c6d', '#2a9d8f', '#e76f51', '#f4a261'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Classification Models Using TFIDF')
plt.show()
```

## State of the Tweet Analysis
### Description
The 'State of Tweet' column indicates the nature of each social media post, as determined by an agreement between annotators.

### Analysis
The distribution of different states of tweets in the dataset is as follows:
- Rumor Posts (r): 1020
- Anti-Rumor Posts (a): 3024
- Question Posts (q): 49
- Non-related Posts (n): 4517

## How to Run the Code
To run the code on Google Colab, follow these steps:
1. Upload the dataset (DATASET_R1.xlsx) to your Google Drive.
2. Open the notebook file (Social_Media_Rumour_Detection.ipynb) in Google Colab.
3. Mount your Google Drive by running the code cell containing the appropriate command.
4. Modify the file path in the code to point to the dataset in your Google Drive.
5. Execute each code cell sequentially to preprocess the data, train the models, and evaluate their performance.

## Conclusion
In conclusion, this project demonstrates the effectiveness of machine learning techniques in detecting rumors in social media posts. By preprocessing text data, vectorizing it, and training classification models, we can accurately classify social media posts as rumors or non-rumors.
