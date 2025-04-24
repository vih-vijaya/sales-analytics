import pandas as pd
import pickle

# 1. Load the new incoming customer data (for prediction)
df = pd.read_csv("data/customer_campaign_data.csv")

# 2. Feature engineering (same as train_model.py)
df['monetary'] = df['total_orders'] * df['avg_order_value']
df['engagement_score'] = df['email_opened'] + df['clicked']

# 3. One-hot encode 'region' (ensure same columns as during training)
df = pd.get_dummies(df, columns=['region'], drop_first=False)

# 4. Handle missing dummy columns (if any region was missing in this batch)
expected_regions = ['region_East', 'region_North', 'region_South', 'region_West']
for col in expected_regions:
    if col not in df.columns:
        df[col] = 0  # Add missing region with 0s

# 5. Prepare feature set (exclude non-feature columns)
X = df.drop(columns=['customer_id', 'converted'], errors='ignore')

# 6. Load the trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# 7. Generate predictions
df['predicted_conversion'] = model.predict(X)

# 8. Save predictions
df[['customer_id', 'predicted_conversion']].to_csv("data/predictions.csv", index=False)

print("Predictions saved to data/predictions.csv")
