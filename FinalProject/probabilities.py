import numpy as np
import main

coviddata = main.makeFileGraph("./data/latestdata.csv")


# wuhan_data_matrix = coviddata[['lives_in_Wuhan', 'reported_market_exposure', 'date_onset_symptoms', 'travel_history_location']].copy()
# wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(axis=1) < 3]
# wuhan_data_matrix['been_in_wuhan'] = np.where(
#     (wuhan_data_matrix['lives_in_Wuhan'] == 1.0) |
#     (wuhan_data_matrix['reported_market_exposure'] == 1.0) |
#     (wuhan_data_matrix['travel_history_location'].str.lower().str.contains("wuhan")),
#     1,
#     0
# )
# no_of_entries = wuhan_data_matrix.shape[0]
# print("Total number of entries",no_of_entries)
# visited_wuhan = wuhan_data_matrix[wuhan_data_matrix['been_in_wuhan'] == 1].shape[0]
# print("Visited Wuhan", visited_wuhan)
# print("Probability of Visited Wuhan", visited_wuhan/no_of_entries)
#
# symptoms_after_wuhan = wuhan_data_matrix[(wuhan_data_matrix['been_in_wuhan'] == 1) & (wuhan_data_matrix['date_onset_symptoms'].notna())].shape[0]
# print("Visited Wuhan and have symptoms", symptoms_after_wuhan)
# print("P(visited Wuhan ∩ have symptoms)", symptoms_after_wuhan/no_of_entries)
#
# print("P(having Symptoms| visited Wuhan): ", (symptoms_after_wuhan/no_of_entries) / (visited_wuhan/no_of_entries))


# wuhan_data_matrix = coviddata[['lives_in_Wuhan', 'reported_market_exposure', 'date_onset_symptoms','date_admission_hospital', 'travel_history_location']].copy()
# wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(axis=1) < 3]
# wuhan_data_matrix['been_in_wuhan'] = np.where(
#     (wuhan_data_matrix['lives_in_Wuhan'] == 1.0) |
#     (wuhan_data_matrix['reported_market_exposure'] == 1.0) |
#     (wuhan_data_matrix['travel_history_location'].str.lower().str.contains("wuhan")),
#     1,
#     0
# )
# no_of_entries = wuhan_data_matrix.shape[0]
# print("Total number of entries",no_of_entries)
# visited_wuhan_with_symptoms = wuhan_data_matrix[(wuhan_data_matrix['been_in_wuhan'] == 1)
#                                                 & (wuhan_data_matrix['date_onset_symptoms'].notnull())].shape[0]
# print("Visited Wuhan and shows symptoms", visited_wuhan_with_symptoms)
# p_visited_with_symptoms = visited_wuhan_with_symptoms/no_of_entries
# print("P(Visited Wuhan with symptoms) = ", p_visited_with_symptoms)
#
# # p_visited_with_symptoms_true_patient
# visited_wuhan_with_symptoms_true_patient = wuhan_data_matrix[(wuhan_data_matrix['been_in_wuhan'] == 1)
#                                                              & (wuhan_data_matrix['date_onset_symptoms'].notnull())
#                                                              & (wuhan_data_matrix['date_admission_hospital'].notnull())].shape[0]
# print("True patients that visited Wuhan showing symptoms", visited_wuhan_with_symptoms_true_patient)
# p_visited_with_symptoms_true_patient = visited_wuhan_with_symptoms_true_patient/no_of_entries
# print("P(Visited Wuhan with symptoms and true patient) = ", p_visited_with_symptoms_true_patient)
#
# print("P(true patient| patient visited wuhan & shows symptoms)", p_visited_with_symptoms_true_patient/p_visited_with_symptoms)

# wuhan_data_matrix = coviddata[['lives_in_Wuhan', 'reported_market_exposure', 'date_onset_symptoms','outcome', 'travel_history_location']].copy()
# wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(axis=1) < 3]
# wuhan_data_matrix['been_in_wuhan'] = np.where(
#     (wuhan_data_matrix['lives_in_Wuhan'] == 1.0) |
#     (wuhan_data_matrix['reported_market_exposure'] == 1.0) |
#     (wuhan_data_matrix['travel_history_location'].str.lower().str.contains("wuhan")),
#     1,
#     0
# )
# wuhan_data_matrix = wuhan_data_matrix.dropna(subset=['outcome'])
# no_of_entries = wuhan_data_matrix.shape[0]
# print("Total number of entries",no_of_entries)
#
# no_been_in_wuhan = wuhan_data_matrix[wuhan_data_matrix['been_in_wuhan'] == 1].shape[0]
# print("Total number of been in wuhan", no_been_in_wuhan)
# p_been_in_wuhan = no_been_in_wuhan/no_of_entries
# print("P(Been in wuhan)", p_been_in_wuhan)
# no_dead_and_been_in_wuhan = wuhan_data_matrix[(wuhan_data_matrix['been_in_wuhan'] == 1)
#                                               & (wuhan_data_matrix['outcome'] == "dead")].shape[0]
# print("No died and been to wuhan", no_dead_and_been_in_wuhan)
# p_dead_and_been_in_wuhan = no_dead_and_been_in_wuhan/no_of_entries
# print("P(been_in_wuhan ∩ died): ", p_dead_and_been_in_wuhan)
#
# print("P(died|been_in_wuhan): ", p_dead_and_been_in_wuhan/p_been_in_wuhan)

# wuhan_data_matrix = coviddata[['lives_in_Wuhan', 'reported_market_exposure', 'date_death_or_discharge','outcome' ,'date_admission_hospital', 'travel_history_location']].copy()
# wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(axis=1) < 3]
# wuhan_data_matrix['been_in_wuhan'] = np.where(
#     (wuhan_data_matrix['lives_in_Wuhan'] == 1.0) |
#     (wuhan_data_matrix['reported_market_exposure'] == 1.0) |
#     (wuhan_data_matrix['travel_history_location'].str.lower().str.contains("wuhan")),
#     1,
#     0
# )
# wuhan_data_matrix = wuhan_data_matrix[['been_in_wuhan', 'date_death_or_discharge', 'date_admission_hospital', 'outcome']]
# wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix['outcome'] != "dead"]
# wuhan_data_matrix = wuhan_data_matrix.dropna(subset=['date_death_or_discharge', 'date_admission_hospital'])
# wuhan_data_matrix['recovery_days'] = (wuhan_data_matrix['date_death_or_discharge'] - wuhan_data_matrix['date_admission_hospital']).dt.days
# wuhan_data_matrix = wuhan_data_matrix[['been_in_wuhan', 'recovery_days']]
# wuhan_data_matrix = wuhan_data_matrix[(wuhan_data_matrix['been_in_wuhan'] == 1) &
#                                       (wuhan_data_matrix['recovery_days'] > 0)]
# print(wuhan_data_matrix['recovery_days'].mean())



wuhan_data_matrix = coviddata[['lives_in_Wuhan', 'reported_market_exposure', 'date_death_or_discharge','outcome' ,'date_onset_symptoms', 'travel_history_location']].copy()
wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix[['lives_in_Wuhan', 'reported_market_exposure', 'travel_history_location']].isnull().sum(axis=1) < 3]
wuhan_data_matrix['been_in_wuhan'] = np.where(
    (wuhan_data_matrix['lives_in_Wuhan'] == 1.0) |
    (wuhan_data_matrix['reported_market_exposure'] == 1.0) |
    (wuhan_data_matrix['travel_history_location'].str.lower().str.contains("wuhan")),
    1,
    0
)
wuhan_data_matrix = wuhan_data_matrix[['been_in_wuhan', 'date_death_or_discharge', 'date_onset_symptoms', 'outcome']]
wuhan_data_matrix = wuhan_data_matrix[wuhan_data_matrix['outcome'] != "dead"]
wuhan_data_matrix = wuhan_data_matrix.dropna(subset=['date_death_or_discharge', 'date_onset_symptoms'])
wuhan_data_matrix['recovery_days'] = (wuhan_data_matrix['date_death_or_discharge'] - wuhan_data_matrix['date_onset_symptoms']).dt.days
wuhan_data_matrix = wuhan_data_matrix[['been_in_wuhan', 'recovery_days']]
wuhan_data_matrix = wuhan_data_matrix[(wuhan_data_matrix['been_in_wuhan'] == 1) &
                                      (wuhan_data_matrix['recovery_days'] > 0)]
print(wuhan_data_matrix['recovery_days'].mean())
