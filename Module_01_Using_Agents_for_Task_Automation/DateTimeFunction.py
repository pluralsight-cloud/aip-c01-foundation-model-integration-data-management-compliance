import json
from datetime import datetime


def lambda_handler(event, context):
    current_datetime = datetime.now()
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup", ""),
            "apiPath": event.get("apiPath", ""),
            "httpMethod": event.get("httpMethod", ""),
            "httpStatusCode": 200,
            "responseBody": {
                "application/json": {
                    "body": json.dumps({
                        "date": current_datetime.strftime("%Y-%m-%d"),
                        "time": current_datetime.strftime("%H:%M:%S")
                    })
                }
            }
        }
    }
