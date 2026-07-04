import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, TEST_SIZE, RANDOM_STATE


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    df = df.drop(["PatientID", "DoctorInCharge"], axis=1)

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled