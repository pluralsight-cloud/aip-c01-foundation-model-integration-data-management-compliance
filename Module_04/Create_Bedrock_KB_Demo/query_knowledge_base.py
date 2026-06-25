import json
import os
import boto3

# Create a Bedrock Agent Runtime client to query knowledge bases
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

# Knowledge base configuration
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
MODEL_ID = "us.amazon.nova-2-lite-v1:0"


def lambda_handler(event, context):
    # Get the question from the event or use a default
    question = event.get("question", "How difficult is the AIP-C01 exam?")

    # Call Retrieve & Generate to get a grounded answer with citations
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ID,
            },
        },
    )

    # Extract the generated answer
    answer = response["output"]["text"]

    # Collect citations showing which sources were used
    citations = []
    for citation in response.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            source = (
                ref.get("location", {}).get("webLocation", {}).get("url")
                or ref.get("location", {}).get("s3Location", {}).get("uri")
            )
            citations.append({
                "text_snippet": ref.get("content", {}).get("text", "")[:200],
                "source": source,
            })

    # Return the answer and citations
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Knowledge base query successful",
            "question": question,
            "answer": answer,
            "citations": citations,
        })
    }
