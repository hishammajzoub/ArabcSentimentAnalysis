# Sentiment Analysis Of Arabic Tweets
#
# Data Set Source: https://www.kaggle.com/datasets/kirolosatef/nlparabictweets
#
# import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import arabic_reshaper
from bidi.algorithm import get_display  # pip install python-bidi
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from colorama import Fore, Style


# Function to tokenize the tweets
def get_words(text):
    tokens = word_tokenize(text)  # split text into words
    return tokens


# init
show_graphs = True
target = "Sentiment"  # target column
POSITIVE = "pos"
NEGATIVE = "neg"
NEUTRAL = "neu"

# read the normalized Excel file
data = pd.read_excel("Normalized Tweets.xlsx")

# drop empty rows
data = data.dropna()

# display program header
print("\n -----------------------------------")
print(" Sentiment Analysis Of Arabic Tweets")
print(" -----------------------------------")

# determine the number of tweets and their percentages for each sentiment
# then, display these values on the screen
pos = data[data[target] == POSITIVE]
neg = data[data[target] == NEGATIVE]
neu = data[data[target] == NEUTRAL]
n_pos = len(pos)
n_neg = len(neg)
n_neu = len(neu)
n_tweets = n_pos + n_neg + n_neu
p_pos = round((n_pos * 100) / n_tweets, 2)
p_neg = round((n_neg * 100) / n_tweets, 2)
p_neu = round((n_neu * 100) / n_tweets, 2)

print("\n ", f"{n_tweets: ,}", "Tweets")
print("\t", n_pos, "Positive Tweets (" + str(p_pos) + "%)")
print("\t", n_neg, "Negative Tweets (" + str(p_neg) + "%)")
print("\t", n_neu, "Neutral Tweets  (" + str(p_neu) + "%)")

# define class labels and their sizes
labels = "Positive", "Negative", "Neutral"
sizes = [n_pos, n_neg, n_neu]

# Create a pie chart to represent the distribution of tweets in relation to sentiments
if show_graphs:
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels)
    plt.title("Distribution Of Sentiments In The Data Set")
    plt.tight_layout()
    plt.show()

# determine features column
features = data.columns.tolist()  # the features columns
features.remove(target)

# split data set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target],
                                                    stratify=data[target], random_state=0, test_size=0.2)
train_data = pd.DataFrame(data=X_train, columns=features)
train_data[target] = y_train
test_data = pd.DataFrame(data=X_test, columns=features)
test_data[target] = y_test

# determine the count as well as the percentage for each data set
n_train_data = len(train_data)
n_test_data = len(test_data)
p_train_data = round(n_train_data * 100 / n_tweets)
p_test_data = round(n_test_data * 100 / n_tweets)
print("\n\t", f"{n_train_data: ,}", "Training Tweets (" + str(p_train_data) + "%)")
print("   \t   ", n_test_data, "Testing Tweets  (" + str(p_test_data) + "%)")

# use TF-IDF to get the training and testing features
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.5, stop_words=None,
                             use_idf=True)
train_data_features = vectorizer.fit_transform(train_data["Text"].values.astype("U"))
test_data_features = vectorizer.transform(test_data["Text"].values.astype("U"))

# group the words for each sentiment
all_positive_words = pos["Text"].apply(get_words)
all_negative_words = neg["Text"].apply(get_words)
all_neutral_words = neu["Text"].apply(get_words)

# flatten the words for each sentiment to find out their frequency distribution
all_positive_words_flat = [word for sublist in all_positive_words for word in sublist]
all_negative_words_flat = [word for sublist in all_negative_words for word in sublist]
all_neutral_words_flat = [word for sublist in all_neutral_words for word in sublist]

positive_frequency_distribution = nltk.FreqDist(all_positive_words_flat)
negative_frequency_distribution = nltk.FreqDist(all_negative_words_flat)
neutral_frequency_distribution = nltk.FreqDist(all_neutral_words_flat)

# sort and reshape the positive words
sorted_positive_words = sorted(positive_frequency_distribution,
                               key=positive_frequency_distribution.get, reverse=True)
positive_words = sorted_positive_words[:15]
positive_counts = [positive_frequency_distribution[word] for word in positive_words]

reshaped_positive_words = [arabic_reshaper.reshape(word) for word in positive_words]
display_positive_words = [get_display(word) for word in reshaped_positive_words]

# display a horizontal bar chart to display the 15 most frequently used positive words
if show_graphs:
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.barh(display_positive_words[::-1], positive_counts[::-1], color="#0cbbf5ff")
    plt.title("Top 15 Words In Positive Tweets", fontsize=18)
    plt.xlabel("Count", fontsize=16)
    plt.ylabel("Words", fontsize=16)
    plt.tight_layout()
    plt.show()

# sort and reshape the negative words
sorted_negative_words = sorted(negative_frequency_distribution,
                               key=negative_frequency_distribution.get, reverse=True)
negative_words = sorted_negative_words[:15]
negative_counts = [negative_frequency_distribution[word] for word in negative_words]

reshaped_negative_words = [arabic_reshaper.reshape(word) for word in negative_words]
display_negative_words = [get_display(word) for word in reshaped_negative_words]

# display a horizontal bar chart to display the 15 most frequently used negative words
if show_graphs:
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.barh(display_negative_words[::-1], negative_counts[::-1], color="#0cbbf5ff")
    plt.title("Top 15 Words In Negative Tweets", fontsize=18)
    plt.xlabel("Count", fontsize=16)
    plt.ylabel("Words", fontsize=16)
    plt.tight_layout()
    plt.show()

# sort and reshape the neutral words
sorted_neutral_words = sorted(neutral_frequency_distribution,
                              key=neutral_frequency_distribution.get, reverse=True)
neutral_words = sorted_neutral_words[:15]
neutral_counts = [neutral_frequency_distribution[word] for word in neutral_words]

reshaped_neutral_words = [arabic_reshaper.reshape(word) for word in neutral_words]
display_neutral_words = [get_display(word) for word in reshaped_neutral_words]

# display a horizontal bar chart to display the 15 most frequently used neutral words
if show_graphs:
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.barh(display_neutral_words[::-1], neutral_counts[::-1], color="#0cbbf5ff")
    plt.title("Top 15 Words In Neutral Tweets", fontsize=18)
    plt.xlabel("Count", fontsize=16)
    plt.ylabel("Words", fontsize=16)
    plt.tight_layout()
    plt.show()

# create classifiers
classifiers = []
classifiers.append(("RF", "Random Forest", RandomForestClassifier(n_estimators=1000, random_state=42)))
classifiers.append(("CART", "Decision Tree", DecisionTreeClassifier()))
classifiers.append(("SVM", "Support Vector Machine", SVC(kernel="linear")))
classifiers.append(("KNN", "K-Nearest Neighbors", KNeighborsClassifier()))
classifiers.append(("MNB", "Multinomial Naive Bayes", MultinomialNB()))
classifiers.append(("LR", "Logistic Regression", LogisticRegression(max_iter=4000)))
classifiers.append(("MLP", "Multi-Layer Perceptron",
                    MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20), random_state=42, verbose=False)))

# process each classifier
print(Fore.BLUE + "\n Evaluating Different Classifiers. Please Wait...", Fore.BLACK)
results = []
for code, name, classifier in classifiers:
    # train/fit the classifier using the training data set
    classifier.fit(train_data_features, train_data[target])

    # predict the response for test data set
    y_predicted = classifier.predict(test_data_features)

    # determine the model's test accuracy score: how often is the classifier correct?
    test_accuracy_score = accuracy_score(y_true=test_data[target], y_pred=y_predicted)

    # store the result in the results list
    results.append((code, name, classifier, test_accuracy_score))

# sort results in descending order of the 4th element (test accuracy score)
results.sort(key=lambda i: i[3], reverse=True)

print("\n #   Classifier \t\t\t\t Test Accuracy Score (%)")
print(" -------------------------------------------------------")
i = 1
for r in results:
    s = r[0] + " - " + r[1]
    print("", i, " ", s, (34 - len(s)) * " ", round(r[3] * 100, 2))
    i += 1
print()

# determine the best classifier that has the highest test accuracy score
# the best classifier is located on the top of the sorted list named results
best_classifier = results[0][2]
best_classifier_code = results[0][0]
best_classifier_name = results[0][1]
print(Fore.MAGENTA, "\n Best Classifier:", best_classifier_code, "-", best_classifier_name,
      "(Test Accuracy Score: " + str(round(results[0][3] * 100, 2)) + "%)")
print(Style.RESET_ALL)

# classify tweets input from the keyboard
while True:
    input_tweet = input(" ? Tweet To Classify: ")
    try:
        input_vectorized = vectorizer.transform([input_tweet])
        prediction = best_classifier.predict(input_vectorized)
        if prediction[0] == POSITIVE:
            print(Fore.GREEN + "\t ==> Predicted Sentiment: Positive")
        elif prediction[0] == NEGATIVE:
            print(Fore.RED + "\t ==> Predicted Sentiment: Negative")
        else:
            print(Fore.CYAN + "\t ==> Predicted Sentiment: Neutral")
        print(Style.RESET_ALL)
    except ValueError:
        print("Invalid input. Please enter a valid tweet!")
