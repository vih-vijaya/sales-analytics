from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the expected region columns
EXPECTED_REGIONS = ['region_West', 'region_East', 'region_South', 'region_North']

class CustomerInput(BaseModel):
    total_orders: int
    avg_order_value: float
    recency: int
    email_opened: int
    clicked: int
    region_West: int
    region_East: int
    region_South: int
    region_North: int

@app.post("/predict")
def predict(input_data: CustomerInput):
    # Feature Engineering
    monetary = input_data.total_orders * input_data.avg_order_value
    engagement_score = input_data.email_opened + input_data.clicked

    # Arrange features as per training order
    feature_list = [
        input_data.total_orders,
        input_data.avg_order_value,
        input_data.recency,
        input_data.email_opened,
        input_data.clicked,
        monetary,
        engagement_score,
        input_data.region_West,
        input_data.region_East,
        input_data.region_South,
        input_data.region_North
    ]

    # Add the missing dummy column (if the model expects it)
    if model.n_features_in_ > len(feature_list):
        print("Adding missing feature column...")
        feature_list.append(0)  # Add dummy 0 if needed

    # Convert to NumPy array
    data = np.array([feature_list])

    # Debugging
    print("Model expects:", model.n_features_in_)
    print("Input shape:", data.shape)

    # Predict
    prediction = model.predict(data)[0]
    return {"converted": bool(prediction)}
