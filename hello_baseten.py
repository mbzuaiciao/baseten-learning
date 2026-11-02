
import os

# Load environment variables from the local .env file.
from dotenv import load_dotenv

# Import the OpenAI client.
# Baseten supports an OpenAI-compatible API,
# so we can use the standard OpenAI Python SDK.
from openai import OpenAI


# 1. Load the API key from .env
# Example .env entry:
# BASETEN_API_KEY=your_api_key_here
load_dotenv()


# 2. Configure the API client
client = OpenAI(
    # Retrieve the API key from the environment.
    # Raises KeyError if BASETEN_API_KEY is missing.
    api_key=os.environ["BASETEN_API_KEY"],

    # Send requests to Baseten instead of OpenAI.
    base_url="https://inference.baseten.co/v1",

    # Maximum time to wait for an API request.
    timeout=60.0,

    # Disable automatic retries while learning.
    # This prevents unexpected additional API calls.
    max_retries=0,
)


# 3. Send a prompt to the hosted model
response = client.chat.completions.create(
    # Model hosted by Baseten.
    # No local model download or GPU is required.
    #model="meta-models/muse-spark-open",
    model="meta-models/muse-spark-open-1026",

    # Conversation messages sent to the model.
    # Each message has a role and content.
    messages=[
        {
            "role": "user",
            "content": (
                "Explain the difference between "
                "training and inference in three sentences."
            ),
        }
    ],

    # Limit the number of generated tokens.
    # This helps control response length and cost.
    reasoning_effort="low",
    max_tokens=1000,
)


# 4. Extract and display the model's answer
# choices[0] selects the first generated response.
# message.content contains the generated text.
print("Model response:")
print(response.choices[0].message.content)


# 5. Display token usage
# This reports tokens consumed by this API request,
# NOT your remaining Baseten promotional credits.
print("\nToken usage:")
print(response.usage)
