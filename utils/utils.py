import json
from pathlib import Path

import requests


def get_llm_response(test_data):
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": test_data["question"],
                                     "chat_history": [
                                     ]
                                 })
    return responseDict


def load_test_data(filename):
    project_directory = Path(__file__).parent.absolute()
    test_data_path = project_directory/"test_data"/filename
    with open(test_data_path, "r") as f:
        return json.load(f)