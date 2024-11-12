from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# Function to load the projected data matrix from Part A
def load_projected_data():
    projected_data = np.load('projected_data.npy')
    return projected_data


# Function to use KNN for all 10 classifications, k-value is 31 since sqrt(1000) ~ 31
# sqrt(n), n being total number of datapoints for the k value is the base standard of KNN
def knn_classifier_all_classes(projected_data, y, k=31):

    X_train, X_test, y_train, y_test = train_test_split(projected_data, y, test_size=0.08, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    predicted_labels = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predicted_labels)

    return accuracy


# Function to use KNN for all only two digits classifications,
# k-value is 31 since sqrt(200) ~ 15
def knn_classifier_two_digits(projected_data, y, digit1, digit2, k=15):

    indices_digit1 = np.where(y == digit1)[0]
    indices_digit2 = np.where(y == digit2)[0]

    selected_indices = np.concatenate([indices_digit1, indices_digit2])

    X_selected = projected_data[selected_indices]
    y_selected = y[selected_indices]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    predicted_labels = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predicted_labels)

    return accuracy


projected_data = load_projected_data()
y = np.repeat(np.arange(10), 100)

accuracy_score_all_classes = knn_classifier_all_classes(projected_data, y)

print("The accuracy of KNN when classifying all the datapoints is", accuracy_score_all_classes)

digit1 = 0
digit2 = 4

accuracy_score_two_digits = knn_classifier_two_digits(projected_data, y, digit1, digit2)

print("The accuracy of KNN when classifying only digits {} and {} is: {}".format(digit1, digit2,
                                                                                 accuracy_score_two_digits))
