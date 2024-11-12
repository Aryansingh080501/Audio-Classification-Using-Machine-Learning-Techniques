import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Function to load the projected data matrix from Part A
def load_projected_data():
    projected_data = np.load('projected_data.npy')
    return projected_data

def load_matrix_U_data():
    matirx_U = np.load('matrix_U.npy')
    return matirx_U
# Function to do SVM for all 10 digits
def svm_classification_all_digits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

# Function to do SVM for only 2 digits
def svm_classification_two_digits(features, labels, digit1, digit2):
    # Select data corresponding to the two digits
    mask = (labels == digit1) | (labels == digit2)
    selected_features = features[mask]
    selected_labels = labels[mask]

    X_train, X_test, y_train, y_test = train_test_split(selected_features, selected_labels, test_size=0.2,
                                                        random_state=42)

    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    report = classification_report(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, report


X = load_projected_data() # the principal components calculated from Part A
y = np.repeat(np.arange(10), 100)  # the labels for the 1000 samples (10 classifications * 100 samples)
'''
    We noticed an abnormally low accuracy rate when classifying all spoken digits. We also included doing SVM
    on just two spoken digits to see if the accuracy rate increased which it did.  

'''
accuracy_score_all, report_all = svm_classification_all_digits(X,y)

print("Accuracy score when classifying all digits with SVM: ", accuracy_score_all)
print("Classification Report when classifying all digits with SVM:")
print(report_all)

digit_1 = 0
digit_2 = 4
accuracy_score_two_digits, report_two_digits = svm_classification_two_digits(X,y,digit_1,digit_2)

print("Accuracy score when classifying {} and {} with SVM : {}".format(digit_1,digit_2,accuracy_score_two_digits))

print("Classification Report when classifying {} and {} with SVM:".format(digit_1,digit_2))
print(report_two_digits)
