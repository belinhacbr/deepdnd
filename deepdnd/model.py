import sys
import os
from dotenv import load_dotenv

import ollama


load_dotenv()

def prompt(msg="Explain Newton's second law of motion") -> str:
    print("User says:", msg)
    response = ollama.chat(
        model=os.getenv('MODEL'),
        messages=[
            {"role": "user", "content": msg},
        ],
    )
    return response["message"]["content"]

if __name__ == '__main__':
    if len(sys.argv) > 1:
        msg = sys.argv[1]
        print(prompt(msg))
    else:
        print("Using default prompt")
        print(prompt())