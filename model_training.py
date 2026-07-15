import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("TRAINING CREDIT CARD APPROVAL MODEL")
print("="*50)

# Step 1: Load dataset
try:
    df = pd.read_csv('dataset/credit_card.csv')
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ dataset/credit_card.csv not found!")
    print("Creating sample dataset...")
    
    # Create sample dataset
    sample_data = {
        'Age': [35, 28, 42, 25, 50, 32, 38, 27, 45, 30],
        'Annual_Income': [750000, 450000, 950000, 250000, 1200000, 580000, 680000, 390000, 880000, 520000],
        'Employment_Status': ['salaried', 'salaried', 'self_employed', 'student', 'government', 'salaried', 'self_employed', 'salaried', 'government', 'unemployed'],
        'Credit_Score': [780, 650, 820, 580, 850, 690, 710, 620, 740, 600],
        'Debt_Ratio': [25, 35, 15, 45, 10, 32, 25, 38, 20, 42],
        'Education_Level': ['graduate', 'graduate', 'post_graduate', '12th', 'post_graduate', 'graduate', 'graduate', '12th', 'graduate', 'graduate'],
        'Marital_Status': ['married', 'single', 'married', 'single', 'married', 'divorced', 'married', 'single', 'married', 'single'],
        'Home_Ownership': ['owned', 'rented', 'owned', 'with_parents', 'owned', 'rented', 'owned', 'rented', 'owned', 'with_parents'],
        'Car_Ownership': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'Number_of_Dependents': [2, 0, 3, 0, 2, 1, 2, 0, 3, 0],
        'Approved': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(sample_data)
    
    # Create dataset folder
    import os
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/credit_card.csv', index=False)
    print("✅ Sample dataset created!")

# Step 2: Data Cleaning
print("\n🔄 Cleaning data...")
df = df.drop_duplicates()

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print(f"✅ Data cleaned: {df.shape[0]} rows")

# Step 3: Encode categorical variables
print("\n🔧 Encoding categorical variables...")
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
print(f"✅ Encoded {len(label_encoders)} categorical columns")

# Step 4: Prepare features and target
print("\n📊 Preparing features...")
target_col = None
possible_targets = ['Approved', 'Status', 'Approval_Status', 'Target']
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"✅ Features: {X.shape[1]} columns")
print(f"✅ Target: {target_col}")

# Step 5: Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✅ Training: {X_train.shape[0]} samples")
print(f"✅ Testing: {X_test.shape[0]} samples")

# Step 7: Train models
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

# Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"   ✅ Accuracy: {lr_accuracy:.4f}")

# Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"   ✅ Accuracy: {rf_accuracy:.4f}")

# Decision Tree
print("\n3. Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"   ✅ Accuracy: {dt_accuracy:.4f}")

# Step 8: Choose best model
models = {
    'Logistic Regression': lr_accuracy,
    'Random Forest': rf_accuracy,
    'Decision Tree': dt_accuracy
}
best_model_name = max(models, key=models.get)
print(f"\n🏆 Best Model: {best_model_name} with {models[best_model_name]:.4f} accuracy")

# Step 9: Save model
print("\n💾 Saving model...")

# Select best model
if best_model_name == 'Logistic Regression':
    final_model = lr_model
elif best_model_name == 'Random Forest':
    final_model = rf_model
else:
    final_model = dt_model

# Create models folder
import os
os.makedirs('models', exist_ok=True)

# Save model and components
model_data = {
    'model': final_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist(),
    'accuracy': models[best_model_name],
    'model_name': best_model_name
}

with open('models/credit_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Model saved to 'models/credit_model.pkl'")
print("\n" + "="*50)
print("🎉 TRAINING COMPLETE!")
print("="*50)
print(f"\n📊 Model Performance Summary:")
print(f"   Logistic Regression: {lr_accuracy:.2%}")
print(f"   Random Forest: {rf_accuracy:.2%}")
print(f"   Decision Tree: {dt_accuracy:.2%}")
print(f"\n🏆 Best: {best_model_name} with {models[best_model_name]:.2%}")
print("\nNow run: python app.py")