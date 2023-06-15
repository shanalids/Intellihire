import pickle

import nltk, re, string, random

# nltk resource download to project
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Importing dependencies
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV

# import training data set
# https://www.stackovercloud.com/2019/09/27/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk/
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')


# minimize the words that contain in the sentence
def lemmatize_sentence(tokens):
    # TAG words with NLTK POS tagger : https://www.nltk.org/book/ch05.html
    # https://wordnet.princeton.edu/
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


# remove noice from sentence
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    # Remove unwanted words and symbols
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

        # Word net lemmatizer : https://www.programcreek.com/python/example/81649/nltk.WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        # If the lemmatized tokens are not punctuation and they are not stop words -> add those tokens to the end of cleaned tokens
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# stop unwanted words in english language
stop_words = stopwords.words('english')

positive_List = twitter_samples.tokenized('positive_tweets.json')
negative_List = twitter_samples.tokenized('negative_tweets.json')

positive_final = []
negative_Final = []

# calling remove noise function
for tokens in positive_List:
    positive_final.append(remove_noise(tokens, stop_words))

for tokens in negative_List:
    negative_Final.append(remove_noise(tokens, stop_words))


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


PositiveWordList = get_all_words(positive_final)

freq_dist_pos = FreqDist(PositiveWordList)


# print(freq_dist_pos.most_common(10))


# check null values
def get_words_for_model(Wordlist):
    for tweet_tokens in Wordlist:
        yield dict([token, True] for token in tweet_tokens)


positive_tokens_for_model = get_words_for_model(positive_final)
negative_tokens_for_model = get_words_for_model(negative_Final)

# add positive and negative label
positiveList = [(tweet_dict, "Positive")
                for tweet_dict in positive_tokens_for_model]

negativeList = [(tweet_dict, "Negative")
                for tweet_dict in negative_tokens_for_model]

# Create a single data set with both negative and positive data sets prepared
SingleList = positiveList + negativeList

# combine positive and negative data
random.shuffle(SingleList)

# divide dataset into train and test
train_data = SingleList[:7000]
test_data = SingleList[7000:]

# train the dataset using naivebayes classifier algorithm
classifyData = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifyData, test_data))

pickle.dump(classifyData, open('model.pkl', 'wb'))

# ---> test question set

# custom_text = "Is smoking good when having lung cancer?"
custom_text = "Smoking is a bad habit"

custom_tokens = remove_noise(word_tokenize(custom_text))

# Test print
print(classifyData.classify(dict([token, True] for token in custom_tokens)))
