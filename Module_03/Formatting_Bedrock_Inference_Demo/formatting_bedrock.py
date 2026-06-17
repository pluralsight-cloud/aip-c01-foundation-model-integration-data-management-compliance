import json
import boto3

# Create Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "us.amazon.nova-micro-v1:0"

# Basic unstructured prompt
print("\n--- Basic Unstructured Prompt ---\n")

basic_messages = [
    {
        "role": "user",
        "content": [
            {"text": "List 3 benefits of structuring prompts when using Amazon Bedrock"}
        ],
    }
]

response_basic = client.converse(
    modelId=MODEL_ID,
    messages=basic_messages,
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.3,
    },
)

basic_output = response_basic["output"]["message"]["content"][0]["text"]
print(f"Prompt:   'List 3 benefits of structuring prompts when using Amazon Bedrock'")
print(f"Response:\n{basic_output}\n")


# Structured prompt with system message
print("\n--- Structured Prompt with System Message ---\n")

system_prompt = [
    {
        "text": (
            "You are a concise AWS generative AI developer. "
            "Always respond in valid JSON with this exact schema: "
            '{"benefits": [{"title": "string", "description": "string"}]}. '
            "Do not include any text outside the JSON object."
        )
    }
]

structured_messages = [
    {
        "role": "user",
        "content": [
            {
                "text": (
                    "List exactly 3 benefits of structuring prompts when using Amazon Bedrock. "
                    "Keep each description under 20 words."
                )
            }
        ],
    }
]

response_structured = client.converse(
    modelId=MODEL_ID,
    messages=structured_messages,
    system=system_prompt,
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.1,       # Lower temp for more deterministic responses
        "topP": 0.9,
        "stopSequences": [],
    },
)

structured_output = response_structured["output"]["message"]["content"][0]["text"]
print("System:   'Respond in valid JSON with a specific schema consisting of title and description'")
print(f"Prompt:   'List exactly 3 benefits of structuring prompts when using Amazon Bedrock. Keep descriptions under 20 words.'")
print(f"Response:\n{structured_output}\n")

