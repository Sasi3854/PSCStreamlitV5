#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:32:18 2025

@author: sasidharankumar
"""

import requests
import os
# Replace with your OpenRouter API key
API_KEY = "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"#os.environ["OPEN_ROUTER_KEY"]

url = "https://openrouter.ai/api/v1/chat/completions"


def get_recommendations(prompt):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "openai/chatgpt-4o-latest",  # Claude 4 Sonnet maps to this identifier
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(result["choices"][0]["message"]["content"])
        return result["choices"][0]["message"]["content"]
    else:
        print("Error:", response.status_code, response.text)
        return  response.text
