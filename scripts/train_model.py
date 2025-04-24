import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# 1. Load the data
df = pd.read_csv("data/customer_campaign_data.csv")

# 2. Feature engineering
df['monetary'] = df['total_orders'] * df['avg_order_value']
df['engagement_score'] = df['email_opened'] + df['clicked']

# 3. One-hot encode the 'region' column
df = pd.get_dummies(df, columns=['region'], drop_first=False)  # Keep all regions

# 4. Prepare features and target
X = df.drop(columns=['customer_id', 'converted'])  # Features
y = df['converted']                                # Target

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save the model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
