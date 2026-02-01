import requests
import json
import os

url = 'http://toolbench.cloud:80/virtual'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

if __name__ == "__main__":

    data = {
        "category": "Artificial_Intelligence_Machine_Learning",
        "tool_name": "TTSKraken",
        "api_name": "List Languages",
        "tool_input": '{}',
        "strip": "truncate",
        "toolbench_key": "Gv7iteCqYR91gBz2a0W6SAHyPfQjwoUusINcdDEXFLJmMblOZ5"
    }
    # data = {
    #     "category": "Data",
    #     "tool_name": "Bible Verse of the Day",
    #     "api_name": "Get Bible Verse of the day",
    #     "tool_input": '{}',
    #     "strip": "truncate",
    #     "toolbench_key": "Gv7iteCqYR91gBz2a0W6SAHyPfQjwoUusINcdDEXFLJmMblOZ5"
    # }
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)