from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from pandas import read_csv
import string
import re


def load_dataset(filename):
    # Helper function to load the dataset into lists
    # of reviews and rating classes (-1, 0, 1)
    df = read_csv(filename, delimiter="\t")
    reviews = []
    y = []

    positive = 0
    neutral = 0
    negative = 0

    for row in df.itertuples():
        # Remove anything that is not a word
        review = re.sub(r"[^a-z\s]", " ", row.review.lower())
        review = re.sub(r"\s+", " ", review)
        reviews.append(review)

        if row.rating <= 4:
            y.append(-1)
            negative += 1
        elif row.rating < 7:
            y.append(0)
            neutral += 1
        else:
            y.append(1)
            positive += 1

    tot = positive + neutral + negative
    print("\tpositive: ", round(positive / tot * 100, 1), "%")
    print("\tneutral: ", round(neutral / tot * 100, 1), "%")
    print("\tnegative: ", round(negative / tot * 100, 1), "%")

    return reviews, y


def compute_score(Clf, X_train, y_train, X_test, y_test):
    # 5-fold cross-validation for alpha hyperparameter
    max = 0
    bestAlpha = 0.1
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        clf = Clf(alpha=a)
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train)
        score = sum(scores) / len(scores)
        if score > max:
            max = score
            bestAlpha = a
    print("\talpha: ", bestAlpha)

    # Create classifier
    clf = Clf(alpha=bestAlpha)
    clf.fit(X_train, y_train)

    # Predict and compute accuracy & Kappa
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    k = metrics.cohen_kappa_score(y_test, y_pred)

    print("\tAccuracy: ", round(acc * 100, 2))
    print("\tKohens's Kappa: ", round(k * 100, 2))


if __name__ == "__main__":

    print("\n> Loading train dataset:")
    X_train_raw, y_train = load_dataset("drugsComTrain_raw.tsv")

    print("\n> Loading test dataset:")
    X_test_raw, y_test = load_dataset("drugsComTest_raw.tsv")

    print("\n> Extracting features")
    v = CountVectorizer(max_df=0.8, ngram_range=(1, 3))
    X_train = v.fit_transform(X_train_raw)
    X_test = v.transform(X_test_raw)

    # Doesn't need binary=True in CountVectorized thanks to binarize from BernoulliNB
    print("\n> BernoulliNB:")
    compute_score(BernoulliNB, X_train, y_train, X_test, y_test)

    print("\n> MultinomialNB:")
    compute_score(MultinomialNB, X_train, y_train, X_test, y_test)
