import boto3
import json

def lambda_handler(event, context):

    # -----------------------------------------------
    # STEP 1 - Read the model ID from AppConfig
    # This is the key pattern - model ID lives outside
    # the code entirely, in AppConfig configuration
    # -----------------------------------------------
    appconfig = boto3.client('appconfigdata', region_name='us-east-1')

    # Start a configuration session with AppConfig
    session = appconfig.start_configuration_session(
        ApplicationIdentifier='bedrock-poc',
        EnvironmentIdentifier='dev',
        ConfigurationProfileIdentifier='model-config'
    )

    # Fetch the latest configuration value
    response = appconfig.get_latest_configuration(
        ConfigurationToken=session['InitialConfigurationToken']
    )

    # Parse the JSON configuration
    config_data = json.loads(response['Configuration'].read())

    # Extract the model settings from AppConfig
    model_id    = config_data['modelId']
    max_tokens  = config_data.get('maxTokens', 300)
    temperature = config_data.get('temperature', 0.7)

    print(f"Model selected from AppConfig: {model_id}")

    # -----------------------------------------------
    # STEP 2 - Call Bedrock using the model from config
    # No model ID is hardcoded anywhere in this function
    # -----------------------------------------------
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )

    prompt = event.get('prompt', 'Explain what a proof of concept is in 2 bullet points.')

    bedrock_response = bedrock.converse(
        modelId=model_id,
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig={
            'maxTokens': max_tokens,
            'temperature': temperature
        }
    )

    response_text = bedrock_response['output']['message']['content'][0]['text']
    usage = bedrock_response['usage']

    # -----------------------------------------------
    # Return the response including which model was used
    # This proves the model came from AppConfig
    # -----------------------------------------------
    return {
        'statusCode': 200,
        'model_used': model_id,
        'source': 'AppConfig',
        'prompt': prompt,
        'response': response_text,
        'token_usage': {
            'input_tokens': usage['inputTokens'],
            'output_tokens': usage['outputTokens'],
            'total_tokens': usage['totalTokens']
        }
    }
