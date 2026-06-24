import json
import boto3

# Create a Bedrock Runtime client to call embedding models
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")


def lambda_handler(event, context):
    # Get sentences from the event or use defaults covering different semantic themes
    sentences = event.get("sentences", [
        "The cat sat on the warm windowsill and watched the birds outside.",
        "Quantum computing uses qubits to perform calculations exponentially faster than classical computers.",
        "To make a perfect risotto, slowly add warm broth while stirring continuously.",
        "The stock market crashed in 1929, leading to the Great Depression.",
        "She felt a wave of happiness wash over her as she opened the acceptance letter.",
        "The Kubernetes pod failed to schedule due to insufficient memory resources."
    ])

    results = []
    for sentence in sentences:
        # Call the embeddings model with the input text
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inputText": sentence,
                "dimensions": 1024,
                "normalize": True
            })
        )

        # Parse the JSON response from the model
        response_body = json.loads(response["body"].read())

        # Store the embedding vector alongside the original text
        results.append({
            "inputText": sentence,
            "embedding": response_body["embedding"],
            "dimensions": len(response_body["embedding"]),
            "inputTextTokenCount": response_body["inputTextTokenCount"]
        })

    # Return all generated embeddings
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Embeddings generated successfully",
            "sentenceCount": len(results),
            "results": results
        })
    }
