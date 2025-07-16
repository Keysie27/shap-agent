from db.firebase import get_subscription_by_key
from datetime import datetime, timezone
import json
import os

def check_subscription_status(key):
    subscription_data = get_subscription_by_key(key)
    expDate = subscription_data.get("expDate")
    now = datetime.now(timezone.utc)
    
    if expDate is None:
        return None
    else:
        return now < expDate

json_path = 'shap-agent/db/subscription_usage.json'
def load_data():
    if not os.path.exists(json_path):
        return {
            "subscription_key": "",
            "date": None,
            "count": 0
        }
    with open(json_path, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def update_field(field, value = None):
    data = load_data()  
    
    if field == "count":
        data[field] += 1
    elif field == "date" and data["date"] != value:
        data["date"] = value 
        data["count"] = 0
    else:
        data[field] = value
    
    save_data(data)     

    