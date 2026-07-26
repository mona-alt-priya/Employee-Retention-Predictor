import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Dataset Read Panrom
df = pd.read_csv('hr_data.csv')

# Target Column Encoding (Yes -> 1, No -> 0)
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' or x == 1 else 0)

# Text Columns-a Numbers-a Convert Panrom
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Data Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Trained Model
with open('dsp_retention_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders, 'features': list(X.columns)}, f)