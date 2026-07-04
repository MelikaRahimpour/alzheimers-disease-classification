from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def get_logistic_regression():
    return LogisticRegression(max_iter=5000)


def get_naive_bayes():
    return GaussianNB()


def get_decision_tree():
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=RANDOM_STATE
    )