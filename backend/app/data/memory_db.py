from datetime import datetime

HISTORY = []

def add_record(record: dict):
    record["timestamp"] = datetime.now().isoformat()
    HISTORY.append(record)
    return record


def get_history():
    return HISTORY