
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Study_Hours': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 2, 6, 7, 8],
    'Attendance': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 48, 82, 88, 98],
    'Assignment_Marks': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 85, 95, 38, 78, 88, 98],
    'Result': ['Fail', 'Fail', 'Fail', 'Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Assignment_Marks']]
y = df['Result']

print("Original Dataset")
print(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    'Voting Classifier': VotingClassifier(
        estimators=[
            ('lr', make_pipeline(StandardScaler(), LogisticRegression())),
            ('dt', DecisionTreeClassifier(random_state=42))
        ],
        voting='hard'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    'Bagging Classifier': BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [85],
    'Assignment_Marks': [80]
})

print("\nNew Student Predictions")

for name, model in models.items():
    print(name, ":", model.predict(new_student)[0])
