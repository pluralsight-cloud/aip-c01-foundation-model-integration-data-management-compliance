import boto3

def lambda_handler(event, context):

    client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )

    MODEL_ID = 'amazon.nova-micro-v1:0'

    prompt = event.get('prompt', 'Explain what real-time inference is in 3 bullet points.')

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig={
            'maxTokens': 300,
            'temperature': 0.7
        }
    )

    response_text = response['output']['message']['content'][0]['text']
    usage = response['usage']

    return {
        'statusCode': 200,
        'model': MODEL_ID,
        'prompt': prompt,
        'response': response_text,
        'token_usage': {
            'input_tokens': usage['inputTokens'],
            'output_tokens': usage['outputTokens'],
            'total_tokens': usage['totalTokens']
        }
    }
