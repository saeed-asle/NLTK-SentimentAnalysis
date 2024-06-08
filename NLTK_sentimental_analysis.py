import nltk
import re
import string
import random
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import matplotlib.pyplot as plt

# Download necessary NLTK resources
#nltk.download('twitter_samples')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def preprocess_text(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0  and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

# Load positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Tokenize a sample tweet
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
print("Tokenized Tweet:")
print(tweet_tokens[0])

# Create a list of stopwords for the English language
stop_words = stopwords.words('english')

# Clean the first positive tweet and display before and after cleaning
print("\nCleaned Tweet:")
print(preprocess_text(tweet_tokens[0], stop_words))

# Clean all positive and negative tweets
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
positive_cleaned_tokens_list = [preprocess_text(tokens, stop_words) for tokens in positive_tweet_tokens]
negative_cleaned_tokens_list = [preprocess_text(tokens, stop_words) for tokens in negative_tweet_tokens]

# Print an example of a positive tweet before and after cleaning
print("\nOriginal Positive Tweet:")
print(positive_tweet_tokens[500])
print("Cleaned Positive Tweet:")
print(positive_cleaned_tokens_list[500])

# Get the frequency distribution of words in the positive tweets
all_pos_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)
print("\nTop 10 Most Common Words in Positive Tweets:")
print(freq_dist_pos.most_common(10))

# Prepare training and testing datasets
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

# Combine positive and negative datasets and shuffle them
positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]
dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

# Split the dataset into training and testing data
train_data = dataset[:7000]
test_data = dataset[7000:]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_data)

# Evaluate the classifier's accuracy on the test data and display the most informative features
print("\nAccuracy is:", classify.accuracy(classifier, test_data))
print("\nMost Informative Features:")
print(classifier.show_most_informative_features(10))


# Test the classifier with custom tweets and display results with plots
def analyze_custom_tweets(classifier, custom_tweets, actual_labels):
    correct_predictions = 0
    total_predictions = len(custom_tweets)

    for i, custom_tweet in enumerate(custom_tweets):
        custom_tokens = preprocess_text(word_tokenize(custom_tweet))
        sentiment = classifier.classify(dict([token, True] for token in custom_tokens))
        probabilities = classifier.prob_classify(dict([token, True] for token in custom_tokens))

        # Display sentiment and confidence
        print(f"Custom Tweet {i + 1}:", custom_tweet)
        print("Sentiment:", sentiment)
        print("Positive Confidence:", probabilities.prob("Positive"))
        print("Negative Confidence:", probabilities.prob("Negative"))
        print("Actual Label:", actual_labels[i])
        print()

        if sentiment == actual_labels[i]:
            correct_predictions += 1

        # Create a bar plot for confidence scores
        labels = ["Positive", "Negative"]
        confidence_scores = [probabilities.prob("Positive"), probabilities.prob("Negative")]
        x = range(len(labels))

        plt.figure()
        plt.bar(x, confidence_scores, align="center")
        plt.xticks(x, labels)
        plt.xlabel("Sentiment")
        plt.ylabel("Confidence Score")
        plt.title(f"Custom Tweet {i + 1} - Confidence Scores")

        plt.show()

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy on Custom Tweets: {accuracy:.2f}%")

# Test the classifier with custom tweets and actual labels
custom_tweets = [
    "I ordered just once from TerribleCo, they screwed up, never used the app again.",
    'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies',
    'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'
]
actual_labels = ["Negative", "Positive", "Positive"]
analyze_custom_tweets(classifier, custom_tweets, actual_labels)

# Test the classifier with custom tweets
custom_tweet = 'The unpredictable weather can be quite an adventure, making each day unique'
custom_tokens = preprocess_text(word_tokenize(custom_tweet))
print("\nSentiment for Custom Tweet 1:", classifier.classify(dict([token, True] for token in custom_tokens)))

custom_tweet = 'The restaurant had excellent food, but the service was a bit lacking.'
custom_tokens = preprocess_text(word_tokenize(custom_tweet))
print("Sentiment for Custom Tweet 2:", classifier.classify(dict([token, True] for token in custom_tokens)))

custom_tweet = 'Despite the initial setbacks, the project eventually achieved its goals.'
custom_tokens = preprocess_text(word_tokenize(custom_tweet))
print("Sentiment for Custom Tweet 3:", classifier.classify(dict([token, True] for token in custom_tokens)))
