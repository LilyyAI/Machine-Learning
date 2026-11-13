import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 6, 8, 3, 5, 7, 9],
    'Attendance': [50, 60, 65, 70, 80, 85, 90, 95, 55, 68, 82, 92, 58, 75, 88, 98],
    'Marks': [35, 40, 45, 55, 65, 75, 85, 95, 38, 48, 70, 90, 42, 60, 80, 98],
    'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass',
               'Fail', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Marks']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
dt = DecisionTreeClassifier(random_state=42)

voting = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt)],
    voting='hard'
)

voting.fit(X_train, y_train)
voting_pred = voting.predict(X_test)

print("Voting Classifier Accuracy:")
print(accuracy_score(y_test, voting_pred))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

bagging = BaggingClassifier(n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)

print("\nBagging Classifier Accuracy:")
print(accuracy_score(y_test, bagging_pred))

print("\nVoting Classification Report:")
print(classification_report(y_test, voting_pred, zero_division=0))