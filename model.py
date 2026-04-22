import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("loan_data.csv")
data.columns = data.columns.str.strip()
data.ffill(inplace=True)
data["Loan_Status"] = np.where(
    (data["Credit_History"] == 1) & (data["ApplicantIncome"] > 3000),
    'Y',
    'N'
)
#convert into binary 
data["Loan_Status"] = data["Loan_Status"].map({'Y': 1, 'N': 0})


data["Gender"] = data["Gender"].map({'Male': 1, 'Female': 0})
data["Married"] = data["Married"].map({'Yes': 1, 'No': 0})

X = data[[
    "Gender",
    "Married",
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History"
]]

y = data["Loan_Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_scaled, y)
# knn_model=KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_scaled,y)
dt_model=DecisionTreeClassifier(random_state=40)
dt_model.fit(X_scaled,y)


pickle.dump(dt_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model of DT and scaler saved successfully!")