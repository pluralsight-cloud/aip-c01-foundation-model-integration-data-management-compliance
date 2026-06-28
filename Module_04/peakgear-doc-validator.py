import json
import boto3
import logging

# Set up logging so all output appears in CloudWatch Logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create an S3 client to check file metadata
s3 = boto3.client("s3")

# File types that Bedrock Knowledge Bases supports for ingestion
SUPPORTED_TYPES = [".txt", ".pdf", ".md", ".html", ".docx", ".csv"]

# Bedrock KB maximum file size is 50MB — reject anything larger
MAX_FILE_SIZE_BYTES = 52428800


def lambda_handler(event, context):
    # Log the incoming event for debugging in CloudWatch Logs
    logger.info(f"Received event: {json.dumps(event)}")

    # Extract the bucket name and file key from the event payload
    bucket = event["bucket"]
    key = event["key"]

    logger.info(f"Validating file: s3://{bucket}/{key}")

    # Collect any validation failures — we check all rules before returning
    validation_errors = []

    # ----------------------------------------------------------------
    # CHECK 1 — Supported file type
    # Extract the file extension and compare against the supported list
    # ----------------------------------------------------------------
    file_extension = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
    if file_extension not in SUPPORTED_TYPES:
        validation_errors.append(
            f"Unsupported file type: '{file_extension}'. "
            f"Supported types are: {', '.join(SUPPORTED_TYPES)}"
        )
        logger.error(f"FAIL — unsupported file type: {file_extension}")
    else:
        logger.info(f"PASS — file type '{file_extension}' is supported")

    # ----------------------------------------------------------------
    # CHECK 2 — File size
    # Use HeadObject to get file metadata without downloading the file
    # ----------------------------------------------------------------
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        file_size = response["ContentLength"]

        if file_size == 0:
            # Empty files would produce empty embeddings — reject them
            validation_errors.append(f"File is empty: {key}")
            logger.error(f"FAIL — file is empty: {key}")
        elif file_size > MAX_FILE_SIZE_BYTES:
            # Files over 50MB exceed the Bedrock KB ingestion limit
            validation_errors.append(
                f"File exceeds 50MB limit: {key} "
                f"({file_size / 1048576:.1f} MB)"
            )
            logger.error(
                f"FAIL — file too large: {file_size / 1048576:.1f} MB"
            )
        else:
            logger.info(
                f"PASS — file size is valid: {file_size / 1024:.1f} KB"
            )

    except Exception as e:
        # S3 read errors (e.g. wrong bucket name) also fail validation
        validation_errors.append(f"Could not read file from S3: {str(e)}")
        logger.error(f"FAIL — S3 read error: {str(e)}")

    # ----------------------------------------------------------------
    # RESULT — return pass or fail to the calling function
    # ----------------------------------------------------------------
    if validation_errors:
        # One or more checks failed — block the ingestion job
        logger.error(f"Validation FAILED for {key}: {validation_errors}")
        return {
            "statusCode": 400,
            "validated": False,
            "file": key,
            "errors": validation_errors
        }

    # All checks passed — signal to the sync trigger to proceed
    logger.info(f"Validation PASSED for {key} — ready for Bedrock KB sync")
    return {
        "statusCode": 200,
        "validated": True,
        "file": key,
        "size_kb": round(file_size / 1024, 1),
        "file_type": file_extension
    }
