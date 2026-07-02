"""

Demonstrates query expansion using Amazon Bedrock.
Takes a vague user query and uses a foundation model to
generate multiple alternative search queries that cover
the same intent from different angles.

This is the first step in an advanced RAG pipeline —
expanding the query before retrieval improves the chance
of finding all relevant documents, not just those that
happen to use the same wording as the original query.
"""

import boto3
import json

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
REGION = "us-east-1"
MODEL_ID = "amazon.nova-lite-v1:0"

# The original user query — vague and underspecified
USER_QUERY = "best way to learn Python"

# Number of alternative queries to generate
NUM_EXPANSIONS = 5

# ----------------------------------------------------------------
# System prompt that instructs the model to act as a
# query expander. The prompt is carefully structured to:
#   1. Define the task clearly
#   2. Specify the output format (numbered list)
#   3. Prevent the model from adding explanation or preamble
# ----------------------------------------------------------------
SYSTEM_PROMPT = """You are a query expansion assistant. Your job is to
take a user's search query and generate alternative versions of it
that express the same intent using different words, phrases, and
levels of specificity.

Rules:
- Generate exactly {num} alternative queries
- Each query must express the same core intent as the original
- Use different vocabulary, synonyms, and phrasings
- Vary the specificity — some broader, some more specific
- Output a numbered list only — no introduction, no explanation
- Each query on its own line, starting with the number and a period
""".format(num=NUM_EXPANSIONS)

# ----------------------------------------------------------------
# Call Bedrock to expand the query
# ----------------------------------------------------------------
def expand_query(query):
    client = boto3.client("bedrock-runtime", region_name=REGION)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"Expand this search query:\n\n{query}"
                }
            ]
        }
    ]

    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.7
        }
    )

    # Extract the text response
    return response["output"]["message"]["content"][0]["text"]


# ----------------------------------------------------------------
# Parse the numbered list response into a clean list of queries
# ----------------------------------------------------------------
def parse_expansions(response_text):
    lines = response_text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading number and period e.g. "1. " or "1) "
        if line[0].isdigit():
            dot_pos = line.find(".")
            paren_pos = line.find(")")
            if dot_pos > 0:
                line = line[dot_pos + 1:].strip()
            elif paren_pos > 0:
                line = line[paren_pos + 1:].strip()
        if line:
            queries.append(line)
    return queries


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("QUERY EXPANSION DEMO")
    print("Using Amazon Bedrock — " + MODEL_ID)
    print("=" * 60)
    print()
    print(f"Original query:")
    print(f"  \"{USER_QUERY}\"")
    print()
    print("Calling Bedrock to expand the query...")
    print()

    raw_response = expand_query(USER_QUERY)
    expanded_queries = parse_expansions(raw_response)

    print(f"Expanded queries ({len(expanded_queries)} alternatives):")
    print("-" * 60)
    for i, q in enumerate(expanded_queries, 1):
        print(f"  {i}. {q}")
    print()
    print("-" * 60)
    print()
    print("In a RAG pipeline each of these queries would now be")
    print("sent to the vector store independently. The results")
    print("then merged and deduplicated before being passed to")
    print("the foundation model as context.")
    print()
    print("This improves retrieval coverage — documents that use")
    print("different terminology for the same concept are found")
    print("by the alternative queries even when the original")
    print("query would have missed them.")
