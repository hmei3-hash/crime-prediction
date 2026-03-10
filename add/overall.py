import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv(r'C:\CSE163\crime-prediction\Crimes_-_2001_to_Present.csv', nrows=20000)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Hour'] = df['Date'].dt.hour

df['Latitude'] = df['Latitude'].fillna(df['Latitude'].median())
df['Longitude'] = df['Longitude'].fillna(df['Longitude'].median())

for col in ['Location Description', 'Primary Type']:
    df[col] = df[col].fillna(df[col].mode()[0])

df_clean = df.dropna(subset=['Block', 'Hour', 'Arrest', 'Year', 'Domestic', 'Community Area', 'IUCR', 'Location Description'])

df_clean['Arrest'] = df_clean['Arrest'].astype(int)
df_clean['Domestic'] = df_clean['Domestic'].astype(int)

crime_counts = df_clean.groupby(['Block', 'Hour']).size().reset_index(name='crime_count')

block_total_crimes = df_clean.groupby('Block').size().reset_index(name='block_total_crimes')
crime_counts = crime_counts.merge(block_total_crimes, on='Block', how='left')

arrest_rates = df_clean.groupby(['Block', 'Hour'])['Arrest'].mean().reset_index(name='arrest_rate')
crime_counts = crime_counts.merge(arrest_rates, on=['Block', 'Hour'], how='left')

feature_agg = df_clean.groupby(['Block', 'Hour']).agg({
    'Community Area': 'first',
    'IUCR': 'first',
    'Location Description': 'first'
}).reset_index()

crime_counts = crime_counts.merge(feature_agg, on=['Block', 'Hour'], how='left')

le_block = LabelEncoder()
crime_counts['Block_encoded'] = le_block.fit_transform(crime_counts['Block'])

le_community = LabelEncoder()
crime_counts['Community Area_encoded'] = le_community.fit_transform(crime_counts['Community Area'].astype(str))

le_iucr = LabelEncoder()
crime_counts['IUCR_encoded'] = le_iucr.fit_transform(crime_counts['IUCR'])

le_location = LabelEncoder()
crime_counts['Location Description_encoded'] = le_location.fit_transform(crime_counts['Location Description'])

features = ['Block_encoded', 'Hour', 'arrest_rate', 'block_total_crimes',
            'Community Area_encoded', 'IUCR_encoded', 'Location Description_encoded']
X = crime_counts[features]
y = crime_counts['crime_count']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("R2:",test_r2)

impo_factors = pd.DataFrame({
    'feature': ['Block', 'Hour', 'Arrest Rate', 'Block Total Crimes','Community Area', 'IUCR', 'Location Description'],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Random Forest Regression: Crime Count Prediction', fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.barh(impo_factors['feature'], impo_factors['importance'])
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance')
for i, j in enumerate(impo_factors['importance']):
    ax1.text(j, i, f' {j:.4f}', va='center')

ax2 = axes[1]
ax2.scatter(y_test, y_pred_test, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Crime Count')
ax2.set_ylabel('Predicted Crime Count')
ax2.set_title('Actual vs Predicted')

plt.tight_layout()
plt.savefig('rf_crime_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

sample_block = crime_counts['Block'].iloc[0]
sample_hour = 12
sample_arrest_rate = 0.5
sample_total_crimes = block_total_crimes[block_total_crimes['Block'] == sample_block]['block_total_crimes'].values[0]

sample_community = crime_counts[crime_counts['Block'] == sample_block]['Community Area'].iloc[0]
sample_iucr = crime_counts[crime_counts['Block'] == sample_block]['IUCR'].iloc[0]
sample_location = crime_counts[crime_counts['Block'] == sample_block]['Location Description'].iloc[0]

sample_X = pd.DataFrame({
    'Block_encoded': [le_block.transform([sample_block])[0]],
    'Hour': [sample_hour],
    'arrest_rate': [sample_arrest_rate],
    'block_total_crimes': [sample_total_crimes],
    'Community Area_encoded': [le_community.transform([sample_community])[0]],
    'IUCR_encoded': [le_iucr.transform([sample_iucr])[0]],
    'Location Description_encoded': [le_location.transform([sample_location])[0]]
})

prediction = rf_model.predict(sample_X)[0]

print(f"  Predicted Crime Count: {prediction} incidents")

print(f"""
Model: Random Forest Regressor

Key Findings (Influencing Factors):
  1. {impo_factors.iloc[0]['feature']}: {impo_factors.iloc[0]['importance']*100:.1f}%
  2. {impo_factors.iloc[1]['feature']}: {impo_factors.iloc[1]['importance']*100:.1f}%
  3. {impo_factors.iloc[2]['feature']}: {impo_factors.iloc[2]['importance']*100:.1f}%
  4. {impo_factors.iloc[3]['feature']}: {impo_factors.iloc[3]['importance']*100:.1f}%
""")