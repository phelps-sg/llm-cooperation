#!/usr/bin/env python3
""" Script to output all available GPT models """
import os

import openai

if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]
    models = openai.Model.list()
    for model_id in [m["id"] for m in models.data if "gpt" in m["id"]]:
        print(model_id)
