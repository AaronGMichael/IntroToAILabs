from statistics import mode as modeStats

import numpy as np
from scipy.stats import mode, multivariate_normal
from sklearn import datasets, neighbors
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB


def getmode(arr):
    modeArr = []
    for i in range(len(arr)):
        modeArr.append([modeStats(arr[i])])
    return modeArr


def TNN(X, Y, k=1):
    distances = metrics.pairwise.euclidean_distances(X)
    sorted_distance = np.argsort(distances, axis=1)[:, 1:k + 1]
    y_pred = Y[sorted_distance]
    # mode_y = getmode(y_pred)
    mode_y, _ = mode(y_pred, axis=1)
    incorrect = np.sum(Y[:, np.newaxis].flatten() != mode_y)
    error = incorrect / len(Y) * 100
    print("Error %: ", error)
    print("Accuracy %: ", 100 - error)
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    error_knn = 100 - np.mean(cross_val_score(knn, X, Y, scoring='accuracy')) * 100
    print("Scikit Accuracy %: ", 100 - error_knn)

def CBN(X, Y):

    classes = np.unique(Y)  # Get the classes in Y
    count_classes = len(classes)  # Get the number of classes in Y
    count_features = X.shape[1]  # Number of features in the label

    # Calculate a priori probabilities (P(ωk))
    each_class_count = np.bincount(Y)  # count each different value in Y
    priors = each_class_count / len(Y)

    # Calculate barycenters (mean of each class)
    barycenters = np.zeros((count_classes, count_features))
    covariances = []
    for i, class_label in enumerate(classes):
        class_indexes = np.where(Y == class_label)[0] # getting the index for each class
        barycenters[i] = np.mean(X[class_indexes], axis=0) # calculating the mean/center for each class feature
        covariances.append(np.diag(np.var(X[class_indexes], axis=0)))

    # Now to classify each data point
    predictions = np.zeros_like(Y)
    predictions2 = np.zeros_like(Y)
    predictions3 = np.zeros_like(Y)
    for i, x in enumerate(X):
        # Calculate conditional probabilities (P(xi/ωk))
        conditional_probability = np.zeros(count_classes)
        conditional_probability2 = np.zeros(count_classes)
        conditional_probability3 = np.zeros(count_classes)
        for k, class_label in enumerate(classes):
            distance = x - barycenters[k]  # Calculate difference from barycenter
            mean = barycenters[k]
            cov = covariances[k]
            covariance = np.cov(X.T)
            #  using the formula for Mahalanobis distance from a centroid
            conditional_probability[k] = priors[k] * np.prod(np.exp(-0.5 * (distance ** 2) / np.diag(covariance)))
            conditional_probability2[k] = priors[k] * np.prod(np.exp(-0.5 * (distance ** 2)))
            conditional_probability3[k] = priors[k] * np.prod(multivariate_normal.pdf(x, mean, np.diag(cov)))

        # Predict class with the highest conditional probability
        predictions[i] = classes[np.argmax(conditional_probability)]
        predictions2[i] = classes[np.argmax(conditional_probability2)]
        predictions3[i] = classes[np.argmax(conditional_probability3)]
    count_correct = np.sum(Y == predictions)
    print("Accuracy V1 %: ", count_correct*100/Y.shape[0])
    print("Prediction Error V1 %: ", (Y.shape[0] - count_correct)*100/Y.shape[0])
    count_correct2 = np.sum(Y == predictions2)
    print("Accuracy V2 %: ", count_correct2 * 100 / Y.shape[0])
    print("Prediction Error V2 %: ", (Y.shape[0] - count_correct2) * 100 / Y.shape[0])
    count_correct3 = np.sum(Y == predictions3)
    print("Accuracy Gaussian %: ", count_correct3 * 100 / Y.shape[0])
    print("Prediction Error Gaussian %: ", (Y.shape[0] - count_correct3) * 100 / Y.shape[0])

    # Using sklearn Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    # Split data 90/10 train/test
    # Create a Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    # Train the classifier on the split training Data
    gnb.fit(X_train, y_train)
    # Make predictions on the split testing data
    y_pred = gnb.predict(X_test)
    # calculate the accuracy of the model
    print("Accuracy SkLearn %", accuracy_score(y_test, y_pred) * 100)
    return predictions

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    print(CBN(X, Y))
