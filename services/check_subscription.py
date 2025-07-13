from db.firebase import get_subscription_by_key
from datetime import datetime, timezone

def check_subscription_status(key):
    subscription_data = get_subscription_by_key(key)
    expDate = subscription_data.get("expDate")
    now = datetime.now(timezone.utc)
    
    if expDate is None:
        return None
    else:
        return now < expDate
    