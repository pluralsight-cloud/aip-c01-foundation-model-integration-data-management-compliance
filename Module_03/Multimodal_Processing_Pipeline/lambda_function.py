import json
import boto3
import time
from datetime import datetime

# Initialize AWS clients
transcribe_client = boto3.client("transcribe")
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")

# Amazon Nova Lite - fast, cost-effective text model
MODEL_ID = "us.amazon.nova-lite-v1:0"


def lambda_handler(event, context):
    """
    Processes a customer call recording:
    1. Starts a Transcribe job to convert audio → text
    2. Waits for transcription to complete
    3. Sends transcript to Bedrock for summarization
    4. Saves the transcript and summary to S3
    """
    print("=== Call Recording Processing Started ===")

    # Get the S3 bucket and key from the event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    print(f"Processing: s3://{bucket}/{key}")

    # Step 1: Transcribe the audio
    print("Step 1: Transcribing audio...")
    transcript = transcribe_audio(bucket, key)
    print(f"Transcript: {len(transcript)} characters")

    # Step 2: Summarize with Bedrock
    print("Step 2: Summarizing with Bedrock...")
    summary = summarize_call(transcript)
    print(f"Summary: {len(summary)} characters")

    # Step 3: Save results
    print("Step 3: Saving results...")
    output_key = save_results(bucket, key, transcript, summary)
    print(f"Saved to: s3://{bucket}/{output_key}")

    print("=== Processing Complete ===")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Call processed successfully",
            "output_file": f"s3://{bucket}/{output_key}",
        }),
    }


def transcribe_audio(bucket, key):
    """
    Uses Amazon Transcribe to convert audio to text.
    Starts a transcription job and polls until complete.
    """
    job_name = f"call-{int(time.time())}"
    file_uri = f"s3://{bucket}/{key}"

    # Determine media format from file extension
    extension = key.rsplit(".", 1)[-1].lower()
    media_format = extension if extension in ["mp3", "mp4", "wav", "flac"] else "mp3"

    # Start the transcription job, outputting to our own S3 bucket
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": file_uri},
        MediaFormat=media_format,
        LanguageCode="en-US",
        OutputBucketName=bucket,
        OutputKey=f"transcripts/{job_name}.json",
    )

    # Poll until the job completes
    while True:
        response = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]

        if status == "COMPLETED":
            break
        elif status == "FAILED":
            raise Exception(f"Transcription failed: {response}")

        time.sleep(5)

    # Get the transcript from S3 (Transcribe wrote it to our bucket)
    transcript_key = f"transcripts/{job_name}.json"
    transcript_obj = s3_client.get_object(Bucket=bucket, Key=transcript_key)
    transcript_data = json.loads(transcript_obj["Body"].read().decode())

    transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]
    return transcript_text


def summarize_call(transcript):
    """
    Sends the transcript to Amazon Bedrock (Nova Lite) for summarization.
    """
    system_prompt = (
        "You are a customer support analyst. Analyze call transcripts and provide:\n"
        "1. A brief summary of the call\n"
        "2. The customer's issue\n"
        "3. The resolution or outcome\n"
        "4. Any follow-up actions needed\n"
        "5. Customer sentiment (positive/neutral/negative)"
    )

    response = bedrock_client.converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [{"text": f"Transcript:\n{transcript}"}],
            }
        ],
        inferenceConfig={
            "maxTokens": 1024,
            "temperature": 0.2,
        },
    )

    summary = response["output"]["message"]["content"][0]["text"]
    return summary


def save_results(bucket, input_key, transcript, summary):
    """
    Saves the transcript and summary to S3.
    """
    filename = input_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_key = f"transcripts/{filename}_{timestamp}.json"

    output = {
        "source_file": f"s3://{bucket}/{input_key}",
        "processed_at": datetime.now().isoformat(),
        "transcript": transcript,
        "summary": summary,
        "model_used": MODEL_ID,
    }

    s3_client.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(output, indent=2),
        ContentType="application/json",
    )

    return output_key
