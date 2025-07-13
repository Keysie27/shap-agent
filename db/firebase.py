import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta
import pathlib
import os
from dotenv import load_dotenv
import uuid

# Initialize Firebase Admin SDK 
if not firebase_admin._apps:
    cred = credentials.Certificate('shap-agent/db/arduino-367f9-firebase-adminsdk-fbsvc-72f0abfa1f.json')  
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Function to add subscription data
def add_subscription(time):
    if time == "month":
        days = 30
    elif time == "year":
        days = 365
        
    expDate = datetime.now() + timedelta(days=days)
    key = uuid.uuid4().hex

    subscription_ref = db.collection('subscriptions').document()  

    # Set data for the new subscription document
    subscription_ref.set({
        'expDate': expDate,
        'key': key
    })

# fetch in for subscription key in db
def get_subscription_by_key():
    #check .env file for search key
    EXTENSION_ROOT = pathlib.Path(__file__).parent.parent.parent  
    env_path = EXTENSION_ROOT / '.env'
    load_dotenv(dotenv_path=env_path, override=True)
    search_key = os.getenv("API_KEY")
        
    #fetch firebase db
    subscriptions_ref = db.collection('subscriptions')
    query = subscriptions_ref.where('key', '==', search_key).limit(1)
    results = query.get()

    if results:
        for doc in results:
            return doc.to_dict()
    else:
        return None
    
# check if key is not expired
def verify_key(data):
    if data is None:
        return False
    
    now = datetime.now(timezone.utc)
    utcExpDate = data["expDate"]
    valid = utcExpDate > now
    return valid
