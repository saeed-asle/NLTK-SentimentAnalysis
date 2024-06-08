# Sentiment Analysis on Twitter Data using NLTK
Authored by saeed asle

# Description
This project performs sentiment analysis on Twitter data using NLTK (Natural Language Toolkit) in Python.
It utilizes the twitter_samples corpus from NLTK, which contains a collection of positive and negative tweets.

The project includes text preprocessing steps such as tokenization, lemmatization, and removing stopwords.
It then trains a Naive Bayes classifier on the preprocessed tweets to predict sentiment (positive or negative) of custom tweets.

# Features
  * Text preprocessing: Tokenization, lemmatization, and removal of stopwords from Twitter data.
  * Training and testing: Trains a Naive Bayes classifier on preprocessed data and evaluates its accuracy on test data.
  * Custom tweet analysis: Analyzes sentiment of custom tweets and displays confidence scores.
  * Interactive sentiment prediction: Predicts sentiment of custom tweets based on trained classifier.
    
# How to Use
  * Ensure you have the necessary NLTK resources downloaded. You can download them using nltk.download() method or uncomment the necessary lines in the code:
    
        #nltk.download('twitter_samples')
        #nltk.download('punkt')
        #nltk.download('wordnet')
        #nltk.download('averaged_perceptron_tagger')
        #nltk.download('stopwords')
  * Run the provided code to perform sentiment analysis on Twitter data and custom tweets.

# Dependencies
  * nltk: For natural language processing tasks.
  * matplotlib: For plotting confidence scores.
    
# Output
The code outputs the sentiment (positive or negative) and confidence scores for custom tweets.
It also evaluates the accuracy of the classifier on test data and displays the most informative features used for classification.
