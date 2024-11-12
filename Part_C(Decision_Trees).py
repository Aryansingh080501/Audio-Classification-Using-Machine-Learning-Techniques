from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Function to load the projected data matrix from Part A
def load_projected_data():
    projected_data = np.load('projected_data.npy')
    return projected_data

# Function to classify using decision tree for all the digits
def train_evaluate_decision_tree_all_digits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09, random_state=42)


    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report =(classification_report(y_test, y_pred))

    return accuracy, report

# Function to classify using decision tree for only two digits

def train_evaluate_decision_tree_two_digits(X, y, digit1, digit2):
    # Filter the dataset to only include the specified two digits
    mask = (y == digit1) | (y == digit2)
    X_filtered = X[mask]
    y_filtered = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

'''
    We noticed an abnormally low accuracy rate when classifying all spoken digits. We also included doing decision tree
    on just two spoken digits to see if the accuracy rate increased which it did.  

'''
X = load_projected_data()  # the principal components calculated from Part A
y = np.repeat(np.arange(10), 100)  # the labels for the 1000 samples (10 classifications * 100 samples)

accuracy_score_all_digits, report_all_digits = train_evaluate_decision_tree_all_digits(X,y)
print("Accuracy for all digits using Decision Tree: ", accuracy_score_all_digits)
print("Classification Report for all digits using Decision tree:")
print(report_all_digits)

digit_1 = 0
digit_2 = 4
accuracy_score_two_digits, report_two_digits = train_evaluate_decision_tree_two_digits(X,y,digit_1,digit_2)
print("Accuracy for only digits {} and {} using Decision Tree: {}".format(digit_1,digit_2,accuracy_score_two_digits))
print("Classification Report for only digits {} and {} using Decision tree:".format(digit_1,digit_2))
print(report_two_digits)