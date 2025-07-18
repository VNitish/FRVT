import uuid
import threading
from Utilities.database import db
from datetime import datetime

def place_return_value_return_payload(traceID="", errorMessage="", displayMessage="", status=False, data={}, code=""):
    return {
        "data": data,
        "status": status,
        "code": code,
        "message": {
            "displayMessage": displayMessage,
            "errorMessage": errorMessage,
            "traceID": traceID
        }
    }

def create_logs(input_data, response, file_name, ip, method_name, method, module_name, request_path,
                logs_type='access'):
    request_data = {
        "_id": unique_id(),
        "users_id": "",
        "request": input_data,
        "response": response,
        "created_at": current_timestamp(),
        "file_name": file_name,
        "ip": ip,
        "api_name": method_name,
        "module_name": module_name,
        "method": method,
        "request_path": request_path,
        "logs_type": logs_type
    }
    db.logs.insert_one(request_data)
    return request_data

def return_process(data, response, method, path, ip, method_name, file_name, module_name, code):
    # Create logs
    keys_to_remove = ["img_base64", "img1", "img2", "base64_code"]
    sanitized_data = {k: v for k, v in data.items() if k not in keys_to_remove}
    
    thread = threading.Thread(
        target=create_logs,
        kwargs={
            "input_data": sanitized_data,
            "response": response,
            "file_name": file_name,
            "ip": ip,
            "method_name": method_name,
            "method": method,
            "request_path": path,
            "logs_type": "access" if not response.get("is_exception", False) else "error",
            "module_name": module_name
        }
    )
    thread.start()
    
    # Return Message
    return_response = place_return_value_return_payload(
        displayMessage=response.get('message', ""), errorMessage=response.get('errorMessage', ""),
        status=response.get("status", False), traceID="", code=code,
        # status=response.get("status", False), traceID=logs_obj['_id'], code=code,
        data=response.get('data', {}))
    return return_response

def current_timestamp():
    return datetime.now().timestamp()

def unique_id():
    return uuid.uuid4().hex