import main

import numpy as np
import pandas as pd

coviddata = main.makeFileGraph("./data/latestdata.csv")
# print(coviddata[coviddata['outcome'].notnull()].head(n=200).to_string(index=False))
findcorrageoutcome = coviddata.dropna(subset=['age','outcome'])
print("correlation with age",findcorrageoutcome['age'].corr(findcorrageoutcome['outcome']))

findcorronsettoadmission = coviddata.dropna(subset=['date_onset_symptoms','date_admission_hospital','outcome'])
findcorronsettoadmission['days_between_symptoms_to_admission'] = findcorronsettoadmission['date_admission_hospital'] - findcorronsettoadmission['date_onset_symptoms']
print("correlation between symptoms to admission in hospital",findcorronsettoadmission['days_between_symptoms_to_admission'].corr(findcorronsettoadmission['outcome']))


findmarketexposurecorr = coviddata.dropna(subset=['reported_market_exposure','outcome'])
print("correlation between visiting the market",findmarketexposurecorr['reported_market_exposure'].corr(findmarketexposurecorr['outcome']))

wuhan_data_matrix = coviddata[['lives_in_Wuhan', 'reported_market_exposure', 'outcome', 'travel_history_location']].copy().dropna(subset=['outcome'])
wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix.isnull().sum(axis=1) < 3]
wuhan_data_matrix['been_in_wuhan'] = np.where(
    (wuhan_data_matrix['lives_in_Wuhan'] == 1.0) |
    (wuhan_data_matrix['reported_market_exposure'] == 1.0) |
    (wuhan_data_matrix['travel_history_location'].str.lower().str.contains("wuhan")),
    1,
    0
)
# print(wuhan_data_matrix.head(n=200).to_string(index=False))
print("correlation between been in Wuhan",wuhan_data_matrix['been_in_wuhan'].corr(wuhan_data_matrix['outcome']))

dates_to_onset = coviddata[['date_onset_symptoms', 'outcome']].dropna(subset=['date_onset_symptoms', 'outcome'])
# print(dates_to_onset.head(n=200).to_string(index=False))
first_date = pd.to_datetime("01.12.2019", format='%d.%m.%Y', errors='coerce')
dates_to_onset['days_to_onset'] = dates_to_onset['date_onset_symptoms'] - first_date
# print(dates_to_onset.head(n=200).to_string(index=False))
print("correlation between days to onset of symptoms", dates_to_onset['days_to_onset'].corr(dates_to_onset['outcome']))

days_spent_in_hospital = coviddata[['outcome', 'date_death_or_discharge', "date_admission_hospital"]].dropna(subset=['outcome', 'date_death_or_discharge', 'date_admission_hospital'])
days_spent_in_hospital['no_of_days'] = days_spent_in_hospital['date_death_or_discharge'] - days_spent_in_hospital['date_admission_hospital']
# print(days_spent_in_hospital.head(n=200).to_string(index=False))
print("correlation between days spent in hospital", days_spent_in_hospital['no_of_days'].corr(days_spent_in_hospital['outcome']))

gender_to_outcome = coviddata[['sex', 'outcome']].dropna(subset=['outcome', 'sex'])
# # print(gender_to_outcome.head(n=200).to_string(index=False))
print("Gender correlation",gender_to_outcome['sex'].corr(gender_to_outcome['outcome']))

symptoms_to_outcome = coviddata[['outcome', 'symptoms']].dropna(subset=['outcome', 'symptoms'])
# print(symptoms_to_outcome.head(n=200).to_string(index=False))
symptoms_df = symptoms_to_outcome['symptoms'].explode().str.get_dummies().groupby(level=0).sum()
df_encoded = pd.concat([symptoms_to_outcome['outcome'], symptoms_df], axis=1)
correlation_matrix = df_encoded.corr()
outcome_correlations = correlation_matrix['outcome'].drop('outcome')
print(outcome_correlations)
# # show changes to parsing