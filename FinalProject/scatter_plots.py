import matplotlib.pyplot as plt
import main
import pandas as pd

coviddata = main.makeFileGraph("./data/latestdata.csv")

# outcome_mapping = {"discharged": 0, "dead": 1}
# coviddata['outcome_num'] = coviddata['outcome'].map(outcome_mapping)
#
#
# plt.scatter(coviddata['age'], coviddata['outcome_num'], c=coviddata['outcome_num'], cmap='viridis')
# plt.xlabel('Age')
# plt.ylabel('Outcome')
# plt.yticks([0, 1], ['Discharged', 'Dead'])
# plt.title('Scatter plot of Age vs. Outcome')
# plt.show()

# deaths = coviddata[coviddata['outcome'] == 'dead']
#
# # Create a histogram of ages for the dead cases
# plt.hist(deaths['age'].dropna(), bins=20, edgecolor='black')
# plt.xlabel('Age')
# plt.ylabel('Number of Deaths')
# plt.title('Histogram of Age vs. Number of Deaths')
# plt.show()


# age_bins = range(0, 101, 10)  # Bins from 0 to 100 with a bin width of 10 years
# coviddata['age_bin'] = pd.cut(coviddata['age'], bins=age_bins)
#
# # Calculate total cases and deaths per age bin
# cases_per_bin = coviddata.groupby('age_bin').size()
# deaths_per_bin = coviddata[coviddata['outcome'] == 'dead'].groupby('age_bin').size()
#
# # Compute death rate per age bin
# death_rate_per_bin = (deaths_per_bin / cases_per_bin) * 100  # Percentage
#
# # Plot the histogram of death rates
# plt.bar(death_rate_per_bin.index.astype(str), death_rate_per_bin, width=0.9, edgecolor='black')
# plt.xlabel('Age Range')
# plt.ylabel('Death Rate (%)')
# plt.title('Death Rate per Age Range')
# plt.xticks(rotation=45)
# plt.show()


findcorronsettoadmission = coviddata.dropna(subset=['date_onset_symptoms','date_admission_hospital','age'])
findcorronsettoadmission['days_between_symptoms_to_admission'] = (findcorronsettoadmission['date_admission_hospital'] - findcorronsettoadmission['date_onset_symptoms']).dt.days

# plt.scatter(findcorronsettoadmission['days_between_symptoms_to_admission'],findcorronsettoadmission['age'], alpha=0.5, s=2)
# plt.ylabel('Age')
# plt.xlabel('Days Between Symptoms to Admission')
# plt.title('Scatter plot of Age vs Days Between Symptoms to Admission')
# plt.show()


average_days_by_age = findcorronsettoadmission.groupby('age')['days_between_symptoms_to_admission'].mean().reset_index()

# Plot scatter plot of average days vs age
plt.scatter(average_days_by_age['age'], average_days_by_age['days_between_symptoms_to_admission'], alpha=0.5, s=50)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Average Days Between Symptoms to Admission')
plt.title('Average Days Between Symptoms to Admission by Age')

# Show plot
plt.show()
