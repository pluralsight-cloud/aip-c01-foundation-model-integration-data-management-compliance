#!/bin/bash
# Deploy the IAM role for the BedrockDynamicModelHandler Lambda function.
# Run this from the directory containing role.yaml.

aws cloudformation deploy \
  --template-file role.yaml \
  --stack-name role \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
