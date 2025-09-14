#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:32:18 2025

@author: sasidharankumar
"""

import requests

# Replace with your OpenRouter API key
API_KEY = "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"

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
