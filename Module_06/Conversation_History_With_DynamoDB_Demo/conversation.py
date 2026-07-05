import boto3
import json
import uuid
import time

# Initialize clients
bedrock = boto3.client("bedrock-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("ConversationHistory")

MODEL_ID = "us.amazon.nova-2-lite-v1:0"
SESSION_ID = str(uuid.uuid4())
TTL_SECONDS = 24 * 60 * 60  # 24 hours


# Save the full message list to DynamoDB with a TTL for automatic cleanup
def save_history(session_id, messages):
    table.put_item(
        Item={
            "SessionId": session_id,
            "Messages": json.dumps(messages),
            "ExpiresAt": int(time.time()) + TTL_SECONDS,
        }
    )


# Retrieve the stored conversation history for a given session
def load_history(session_id):
    response = table.get_item(Key={"SessionId": session_id})
    if "Item" in response:
        return json.loads(response["Item"]["Messages"])
    return []


# Send the conversation messages to Bedrock and return the model's response text
def call_bedrock(messages):
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 256, "temperature": 0.5},
    )
    return response["output"]["message"]["content"][0]["text"]


# --- Turn 1: Introduce context ---
print("=" * 60)
print("TURN 1 - Establishing context")
print("=" * 60)

user_message_1 = {
    "role": "user",
    "content": [{"text": "My name is Alex and I'm learning about AWS."}],
}
messages = [user_message_1]
print(f"\nUser: {user_message_1['content'][0]['text']}")

response_text = call_bedrock(messages)
print(f"\nAssistant: {response_text}")

# Append the assistant reply to messages
assistant_message_1 = {
    "role": "assistant",
    "content": [{"text": response_text}],
}
messages.append(assistant_message_1)

# Save conversation history to DynamoDB
save_history(SESSION_ID, messages)
print(f"\n>> Conversation saved to DynamoDB (SessionId: {SESSION_ID})")

# --- Turn 2: Retrieve history and ask a follow-up ---
print("\n" + "=" * 60)
print("TURN 2 - Using stored history for context")
print("=" * 60)

# Load history from DynamoDB (simulating a new request)
history = load_history(SESSION_ID)
print(f"\n>> Loaded {len(history)} messages from DynamoDB")
print(">> Injecting history into the new prompt...\n")

# Add the follow-up question
follow_up = {
    "role": "user",
    "content": [{"text": "What is my name and what am I learning about?"}],
}
history.append(follow_up)
print(f"User: {follow_up['content'][0]['text']}")

# Call Bedrock with full history — the model now has context
response_text = call_bedrock(history)
print(f"\nAssistant: {response_text}")

# Save updated history
assistant_message_2 = {
    "role": "assistant",
    "content": [{"text": response_text}],
}
history.append(assistant_message_2)
save_history(SESSION_ID, history)
print(f"\n>> Updated conversation saved to DynamoDB")
print("=" * 60)
