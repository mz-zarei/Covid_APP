# -*- coding: utf-8 -*-
"""
@author: Mohammad Zarei
"""

import requests

IMG_PATH = "/Users/mz/Documents/GitHub_Projects/COVID_APP/2_DeployModel/testImages/positive.jpg"
resp = requests.post("http://127.0.0.1:5000/predict", files={"file": open(IMG_PATH,'rb')})


print(resp.json())

IMG_PATH = "/Users/mz/Documents/GitHub_Projects/COVID_APP/2_DeployModel/testImages/negative.jpg"
resp = requests.post("http://127.0.0.1:5000/predict", files={"file": open(IMG_PATH,'rb')})


print(resp.json())
