import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import main

coviddata = main.makeFileGraph("./data/latestdata.csv")

# coviddata = coviddata.dropna(subset=['age', 'outcome'])
#
# # Encode 'outcome' variable into numerical values
# label_encoder = LabelEncoder()
# coviddata['outcome_encoded'] = label_encoder.fit_transform(coviddata['outcome'])
#
# # Select features for PCA
# features = ['age', 'outcome_encoded']
#
# # Standardize the features
# scaled_features = (coviddata[features] - coviddata[features].mean()) / coviddata[features].std()
#
# # Perform PCA
# pca = PCA(n_components=2)
# projected_data = pca.fit_transform(scaled_features)
#
# # Create a DataFrame for the projected data
# projected_df = pd.DataFrame(data=projected_data, columns=['PC1', 'PC2'])
#
# # Plot the projected data
# plt.scatter(projected_df['PC1'], projected_df['PC2'], c=coviddata['outcome_encoded'], cmap='viridis', alpha=0.5)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Projection of Outcome vs Age')
# plt.colorbar(label='Outcome')
# plt.show()


# Drop rows with missing values in 'age', 'symptoms', and 'outcome'
coviddata = coviddata.dropna(subset=['age', 'outcome'])

# Encode 'outcome' variable into numerical values
label_encoder = LabelEncoder()
coviddata['outcome_encoded'] = label_encoder.fit_transform(coviddata['outcome'])

coviddata['been_in_wuhan'] = np.where(
    # Check if any of the conditions are True
    (coviddata['lives_in_Wuhan'].notnull()) |  # Check if 'lives_in_Wuhan' is not null (has a value)
    (coviddata['reported_market_exposure'].notnull()) |  # Check if 'reported_market_exposure' is not null
    (coviddata['travel_history_location'].notnull()),  # Check if 'travel_history_location' is not null
    # If any condition is True, evaluate the original logic
    np.where(
        (coviddata['lives_in_Wuhan'] == 1.0) |
        (coviddata['reported_market_exposure'] == 1.0) |
        (coviddata['travel_history_location'].str.lower().str.contains("wuhan")),
        1,
        0
    ),
    # If none of the conditions are True (all missing), return None
    None
)

coviddata = coviddata.dropna(subset=['been_in_wuhan'])
coviddata['no_of_days'] = pd.to_timedelta(coviddata['date_death_or_discharge'].dt.date -
                                         coviddata['date_admission_hospital'].dt.date, errors='coerce')

coviddata = coviddata.dropna(subset=['no_of_days'])

# Convert 'symptoms' array into dummy variables using one-hot encoding
symptoms_array = coviddata['symptoms'].apply(lambda x: ','.join(x) if isinstance(x, list) else x).str.get_dummies(sep=',')
# coviddata_encoded = pd.concat([coviddata[['age', 'been_in_wuhan', 'no_of_days']], symptoms_array], axis=1)
coviddata_encoded = pd.concat([coviddata[['age', 'been_in_wuhan', 'no_of_days']]], axis=1)


# Select features for PCA
features = coviddata_encoded.columns

# Standardize the features
scaled_features = (coviddata_encoded - coviddata_encoded.mean()) / coviddata_encoded.std()

# Perform PCA
pca = PCA(n_components=2)
projected_data = pca.fit_transform(scaled_features)

# Create a DataFrame for the projected data
projected_df = pd.DataFrame(data=projected_data, columns=['PC1', 'PC2'])

# Plot the projected data
plt.scatter(projected_df['PC1'], projected_df['PC2'], c=coviddata['outcome_encoded'], cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Outcome from Age and Symptoms')
plt.colorbar(label='Outcome')
plt.show()

# Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_

# Loadings (replace 'features' with actual column names if different)
loadings = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])

# Print Explained Variance Ratio table
print("Explained Variance Ratio:")
print(explained_variance)

# Print Loadings table
print("\nLoadings:")
print(loadings.to_string())
