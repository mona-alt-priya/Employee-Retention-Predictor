import pandas as pd
import numpy as np

# Sample HR Data Generator (500 Employees)
np.random.seed(42)
n_samples = 500

data = {
    'Age': np.random.randint(22, 60, n_samples),
    'MonthlyIncome': np.random.randint(2500, 20000, n_samples),
    'TotalWorkingYears': np.random.randint(1, 30, n_samples),
    'YearsAtCompany': np.random.randint(0, 15, n_samples),
    'YearsSinceLastPromotion': np.random.randint(0, 10, n_samples),
    'JobSatisfaction': np.random.randint(1, 5, n_samples),
    'DistanceFromHome': np.random.randint(1, 40, n_samples),
    'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
}

df = pd.DataFrame(data)
df.to_csv('hr_data.csv', index=False)