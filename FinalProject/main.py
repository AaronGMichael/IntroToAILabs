import pandas as pd

count1 = 0
count2 = 0
countnull = 0


def correctage(x):
    if x == '':
        return None
    age = 0
    try:
        if "-" in x:
            value = x.split("-")
            age1 = float(value[0])
            if (value[1] != ''):
                age2 = float(value[1])
            else:
                age2 = age1
            age = (age1 + age2) / 2
        elif "+" in x:
            value = x.split("+")
            age = float(value[0])
        elif "month" in x or "week" in x:
            value = x.split(" ")
            age = float(value[0])
            if "week" in x:
                age = float(age / 52)
            if "month" in x:
                age = float(age / 12)
        else:
            age = float(x)
    except ValueError:
        print("error: ", x.split("-"))
    return age


def get_boolean(x):
    if x == '' or x.lower() == "na":
        return None
    if x.lower() == "no":
        return 0
    elif (x.lower() == "yes") or "yes" in x.lower():
        return 1
    return 1


def get_outcome(x):
    if x == '' or x.lower() == "na":
        return None
    elif (x.lower() == "died"
          or "death" in x.lower() or
          "deceased" in x.lower() or
          "critical" in x.lower() or
          "severe" in x.lower() or "dead" in x.lower()):
        return "died"
    elif ("discharge" in x.lower()
          or "recovered" in x.lower()
          or "hospitalized" in x.lower()
          or "stable" in x.lower()
          or "alive" in x.lower() or "treat" in x.lower()
          or "releas" in x.lower() or "recover" in x.lower()):
        return "discharged"
    return None


def parse_symptoms(x):
    if x == '' or x.lower() == "na":
        return None
    lowcase = x.lower()
    symptoms = set()
    if "fever" in lowcase or "flu" in lowcase:
        symptoms.add("fever")
    if "cough" in lowcase or "flu" in lowcase:
        symptoms.add("cough")
    if "throat" in lowcase:
        symptoms.add("sore throat")
    if "pneumon" in lowcase or "respiratory" in lowcase or "pnuem" in lowcase or "pulmon" in lowcase or "acute pharyn" in lowcase:
        symptoms.add("pneumonia")
    if "shortness" in lowcase or "chest" in lowcase or "dyspnea" in lowcase or "breathi" in lowcase:
        symptoms.add("breathing trouble")
    if "tired" in lowcase or "fatigue" in lowcase or "malaise" in lowcase or "soren" in lowcase:
        symptoms.add("fatigue")
    if "nausea" in lowcase or "dizz" in lowcase or "headache" in lowcase:
        symptoms.add("nausea")
    if "catarrhal" in lowcase or "covid" in lowcase or "moderate" in lowcase:
        symptoms.update(["fever", "runny nose", "sore throat", "cough"])
    if "severe" in lowcase or "poor" in lowcase or "failure" in lowcase:
        symptoms.add("severe")
    if "run" in lowcase or "rhino" in lowcase or "sputum" in lowcase:
        symptoms.add("runny nose")
    if (len(symptoms) == 0):
        return None
    return list(symptoms)


def parse_sex(x):
    if x == '' or x.lower() == "na":
        return None
    if x.lower() == "male" or x.lower() == "m":
        return 1
    elif x.lower() == "female" or x.lower() == "f":
        return 0
    else:
        return None


def makeFileGraph(textfile, delimiter=","):
    count = 0
    date_columns = ["date_onset_symptoms",
                    "date_admission_hospital",
                    "date_confirmation",
                    "date_death_or_discharge"]
    return pd.read_csv(textfile,
                       delimiter=delimiter,
                       low_memory=False,
                       converters={"age": lambda x: correctage(x),
                                   "lives_in_Wuhan": lambda x: get_boolean(x),
                                   "reported_market_exposure": lambda x: get_boolean(x),
                                   "outcome": lambda x: get_outcome(x),
                                   "symptoms": lambda x: parse_symptoms(x),
                                   "sex": lambda x: parse_sex(x)},
                       parse_dates=date_columns,
                       date_parser=lambda x: pd.to_datetime(x,
                                                            format='%d.%m.%Y',
                                                            errors='coerce')
                       )



