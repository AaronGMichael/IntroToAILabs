import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

import main
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, \
    mean_squared_error, silhouette_score


def calculate_recovery_interval(row):
    discharge_date = row['date_death_or_discharge']
    onset_date = row['date_onset_symptoms']
    if pd.isna(discharge_date):
        return None  # Return None if either is missing
    if pd.isna(onset_date):
        onset_date = row['date_admission_hospital']
    if pd.isna(onset_date):
        return None
    else:
        return (discharge_date - onset_date).days  # Extract the actual value


def random_sample_imputation(df):
    cols_with_missing_values = df.columns[df.isna().any()].tolist()

    for var in cols_with_missing_values:
        # extract a random sample
        random_sample_df = df[var].dropna().sample(df[var].isnull().sum(),
                                                   random_state=0, replace=True)
        # re-index the randomly extracted sample
        random_sample_df.index = df[
            df[var].isnull()].index

        # replace the NA
        df.loc[df[var].isnull(), var] = random_sample_df

    return df

coviddata = main.makeFileGraph("./data/latestdata.csv")
usefuldata = coviddata.drop(
    labels=['ID', 'geo_resolution', 'additional_information', 'source', 'sequence_available', 'notes_for_discussion',
            'admin3', 'admin2', 'admin1', 'admin_id', 'data_moderator_initials', 'travel_history_binary',
            'date_confirmation', 'travel_history_dates'], axis=1)

usefuldata['recovery_interval'] = usefuldata.apply(calculate_recovery_interval, axis=1)
# usefuldata = usefuldata.dropna(subset=['symptoms'])
# symptoms_array = usefuldata['symptoms'].apply(lambda x: ','.join(x) if isinstance(x, list) else x).str.get_dummies(sep=',')
# usefuldata = pd.concat([usefuldata, symptoms_array], axis=1)
# features = ['age', 'been_in_wuhan', 'chronic_disease_binary', 'recovery_interval', "breathing trouble","cough","fatigue","fever","nausea","pneumonia","runny nose","severe","sore throat"]
features = ['age', 'been_in_wuhan', 'chronic_disease_binary', 'recovery_interval']
target = 'outcome'
#
usefuldata = usefuldata[
    usefuldata[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(
        axis=1) < 3]

usefuldata['been_in_wuhan'] = np.where(
    usefuldata[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(
        axis=1) < 3,
    np.where((usefuldata['lives_in_Wuhan'] == 1.0) |
             (usefuldata['reported_market_exposure'] == 1.0) |
             (usefuldata['travel_history_location'].str.lower().str.contains("wuhan")),
             1,
             0), None
)

# print(usefuldata.head(n=100).to_string())

df = usefuldata.dropna(subset=[target])
# df['chronic_disease_binary'] = df['chronic_disease_binary'].fillna(0)
df[target] = df[target].map({'died': 1, 'discharged': 0})
# print(df['recovery_interval'].mode()[0])
# print(df.shape)
# df = random_sample_imputation(df)
df = df.dropna(subset=features)
X = df[features]
y = df[target]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Sample data and target variable (replace with your data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1, 1],
}

# Create the GridSearchCV object
svc_model = SVC()
grid_search = GridSearchCV(svc_model, param_grid, cv=5)  # 5-fold cross-validation

# Fit the GridSearch
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use the best model for prediction
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1score = f1_score(y_test, predictions)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1score:.2f}")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
#
# y_pred = knn.predict(X_test)
#
# conf_matrix = confusion_matrix(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# f1score = f1_score(y_test, y_pred)
#
# print("Confusion Matrix:")
# print(conf_matrix)
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"F1 Score: {f1score:.2f}")
#
# Select relevant features and target variable
# features = ['recovery_interval', 'chronic_disease_binary',  "breathing trouble",
#             "cough","fatigue","fever","nausea","pneumonia","runny nose",
#             "severe","sore throat"]  # Add more relevant features if available
# # features = ['recovery_interval', 'chronic_disease_binary']
# target = 'age'
#
# df = usefuldata.dropna(subset=[target])
#
# df = df.dropna(subset=features)
#
# X = df[features]
# y = df[target]
#
# numeric_features = ['recovery_interval', 'chronic_disease_binary',  "breathing trouble","cough","fatigue","fever","nausea","pneumonia","runny nose","severe","sore throat"]
# # numeric_features = ['recovery_interval', 'chronic_disease_binary']
# numeric_transformer = StandardScaler()
#
# categorical_features = []
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])
#
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', LinearRegression())])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# pipeline.fit(X_train, y_train)
#
# y_pred = pipeline.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred)
#
# print(f"Mean Squared Error: {mse:.2f}")
#
# features = ['age', 'recovery_interval', 'been_in_wuhan', 'chronic_disease_binary']
# target = 'outcome'
#
# # Drop rows with missing target values
# df = usefuldata.dropna(subset=[target])
#
# # Encode target variable
# df[target] = df[target].map({'died': 1, 'discharged': 0})
#
# # Handle missing values in features (for simplicity, we'll drop rows with missing values)
# df = df.dropna(subset=features)
#
# # Separate majority and minority classes
# df_majority = df[df[target] == 0]
# df_minority = df[df[target] == 1]
#
#
# # Downsample majority class
# df_majority_downsampled = resample(df_majority,
#                                    replace=False,    # sample without replacement
#                                    n_samples=df_minority.shape[0],  # to match minority class
#                                    random_state=42)  # reproducible results
#
# # Combine minority class with downsampled majority class
# df_balanced = pd.concat([df_majority_downsampled, df_minority])
#
# # Separate features and target
# X_balanced = df_balanced[features]
# y_balanced = df_balanced[target]
#
# # Split the balanced data into training and testing sets
# X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
#
# # Standardize the features
# scaler = StandardScaler()
# X_train_balanced = scaler.fit_transform(X_train_balanced)
# X_test_balanced = scaler.transform(X_test_balanced)
#
# # Train the K-NN classifier on the balanced data
# knn_balanced = KNeighborsClassifier(n_neighbors=5)
# knn_balanced.fit(X_train_balanced, y_train_balanced)
#
# # Make predictions
# y_pred_balanced = knn_balanced.predict(X_test_balanced)
#
# # Evaluate the model on the balanced data
# conf_matrix_balanced = confusion_matrix(y_test_balanced, y_pred_balanced)
# accuracy_balanced = accuracy_score(y_test_balanced, y_pred_balanced)
# recall_balanced = recall_score(y_test_balanced, y_pred_balanced)
# precision_balanced = precision_score(y_test_balanced, y_pred_balanced)
# f1_balanced = f1_score(y_test_balanced, y_pred_balanced)
#
# print("Confusion Matrix (Balanced):")
# print(conf_matrix_balanced)
# print(f"Accuracy (Balanced): {accuracy_balanced:.2f}")
# print(f"Recall (Balanced): {recall_balanced:.2f}")
# print(f"Precision (Balanced): {precision_balanced:.2f}")
# print(f"F1 Score (Balanced): {f1_balanced:.2f}")
#



# features = ['age', 'recovery_interval', 'outcome']
#
# # Handle missing values in features
# df = usefuldata.dropna(subset=features)
#
# # Filter out non-positive recovery intervals
# df = df[df['recovery_interval'] > 0]
#
# # Separate features
# X = df[features]
#
# # Preprocessing for numerical data
# numeric_features = ['age', 'recovery_interval', 'outcome']
# numeric_transformer = StandardScaler()
#
# # Create a preprocessor
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features)
#     ])
#
# # Create a pipeline
# pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
#
# # Preprocess the data
# X_preprocessed = pipeline.fit_transform(X)
#
# # Determine the optimal number of clusters using the Silhouette index
# silhouette_scores = []
# range_n_clusters = list(range(2, 11))
#
# for n_clusters in range_n_clusters:
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(X_preprocessed)
#     silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
#     silhouette_scores.append(silhouette_avg)
#
# # Plot Silhouette scores
# plt.figure(figsize=(10, 6))
# plt.plot(range_n_clusters, silhouette_scores, marker='o')
# plt.title("Silhouette Scores for Different Numbers of Clusters")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Score")
# plt.show()
#
# # Find the optimal number of clusters
# optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
# print(f"The optimal number of clusters is: {optimal_n_clusters}")
#
# # Apply K-means with the optimal number of clusters
# kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(X_preprocessed)
#
# # Add the cluster labels to the original dataframe
# df['cluster'] = cluster_labels
#
# # Visualize the clustering structure
# plt.figure(figsize=(10, 6))
# plt.scatter(df['age'], df['recovery_interval'], c=df['cluster'], cmap='viridis')
# plt.title("Clusters of Persons Based on Age, Recovery Interval and outcome")
# plt.xlabel("Age")
# plt.ylabel("Recovery Interval")
# plt.colorbar(label='Cluster')
# plt.show()


