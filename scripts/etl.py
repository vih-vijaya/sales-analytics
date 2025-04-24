import pandas as pd

df = pd.read_csv("data/customer_campaign_data.csv")

# Feature Engineering
df['monetary'] = df['total_orders'] * df['avg_order_value']
df['engagement_score'] = df['email_opened'] + df['clicked']
df['region'] = df['region'].astype('category')
df = pd.get_dummies(df, columns=['region'])

df.to_csv("data/cleaned_data.csv", index=False)
