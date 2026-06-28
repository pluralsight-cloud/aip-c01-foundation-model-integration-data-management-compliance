import json
import boto3
import logging

# Set up logging so all output appears in CloudWatch Logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Replace this with your actual Knowledge Base ID from the Bedrock console
KNOWLEDGE_BASE_ID = "YOUR_KB_ID_HERE"

# Create a Bedrock Agent client to call the ingestion job API
bedrock_agent = boto3.client("bedrock-agent", region_name="us-east-1")

# Create a Lambda client to invoke the validator function
lambda_client = boto3.client("lambda", region_name="us-east-1")


def get_data_source_id():
    # Look up the data source ID dynamically rather than hardcoding it
    # This means the function keeps working if the data source is recreated
    response = bedrock_agent.list_data_sources(
        knowledgeBaseId=KNOWLEDGE_BASE_ID
    )
    return response["dataSourceSummaries"][0]["dataSourceId"]


def validate_document(bucket, key):
    # Invoke the validator Lambda synchronously and wait for the result
    # InvocationType RequestResponse means we block until it returns
    response = lambda_client.invoke(
        FunctionName="peakgear-doc-validator",
        InvocationType="RequestResponse",
        Payload=json.dumps({"bucket": bucket, "key": key})
    )
    # Parse the response payload from bytes to a Python dictionary
    result = json.loads(response["Payload"].read())
    logger.info(f"Validation result for {key}: {result}")

    # Return True if validated is True, False for anything else
    return result.get("validated", False)


def lambda_handler(event, context):
    # Log the full SQS event for debugging in CloudWatch Logs
    logger.info(f"Received event: {json.dumps(event)}")

    # Get the Bedrock data source ID before processing any records
    data_source_id = get_data_source_id()
    logger.info(f"Data source ID: {data_source_id}")

    # SQS can deliver multiple messages in one invocation — process each one
    for record in event["Records"]:

        # Parse the SQS message body — this contains the EventBridge event
        body = json.loads(record["body"])

        # Extract the S3 bucket name and object key from the EventBridge event
        bucket = body.get("detail", {}).get("bucket", {}).get("name", "")
        key = body.get("detail", {}).get("object", {}).get("key", "")

        logger.info(f"File uploaded: s3://{bucket}/{key}")

        # Step 1 — Validate the file before triggering an expensive sync
        if not validate_document(bucket, key):
            # Validation failed — log and skip to the next SQS message
            logger.error(f"Validation FAILED for {key} — skipping sync")
            continue

        logger.info(f"Validation PASSED for {key} — triggering sync")

        # Step 2 — Start the Bedrock Knowledge Base ingestion job
        try:
            response = bedrock_agent.start_ingestion_job(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=data_source_id
            )
            # Log the job ID so we can track it in the Bedrock console
            job_id = response["ingestionJob"]["ingestionJobId"]
            logger.info(f"Ingestion job started: {job_id}")

        except Exception as e:
            if "ConflictException" in str(e):
                # A job is already running for this data source
                # This is expected if multiple files are uploaded at once
                # The SQS queue will retry this message after visibility timeout
                logger.warning(
                    "Ingestion job already in progress — skipping"
                )
            else:
                # Unexpected error — re-raise so SQS retries the message
                logger.error(f"Failed to start ingestion job: {str(e)}")
                raise

    return {"statusCode": 200, "body": "Pipeline completed"}
