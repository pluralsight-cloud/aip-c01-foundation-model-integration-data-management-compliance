import json
import sys
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Configuration
REGION = "us-east-1"
INDEX_NAME = "support-docs"
MODEL_ID = "amazon.titan-embed-text-v2:0"
TOP_K = 3

# Get the collection endpoint and query from the command line
if len(sys.argv) < 3:
    print("Usage: python3 search.py <collection-endpoint> <query>")
    print('Example: python3 search.py https://abc123.us-east-1.aoss.amazonaws.com "How do I return an item?"')
    sys.exit(1)

endpoint = sys.argv[1]
query = sys.argv[2]
host = endpoint.replace("https://", "")

# --- AWS authentication ---
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "aoss",
    session_token=credentials.token
)

# --- OpenSearch client ---
os_client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

# --- Bedrock client for embeddings ---
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def get_embedding(text):
    """Convert text into a vector embedding using Titan Text Embeddings V2."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })
    )
    return json.loads(response["body"].read())["embedding"]


# === Step 1: Convert the query into an embedding ===
print(f'\nQuery: "{query}"')
print("Converting query to embedding...")
query_embedding = get_embedding(query)

# Display the embedding vector (truncated for readability)
print(f"\nQuery embedding ({len(query_embedding)} dimensions):")
print(f"  [{query_embedding[0]:.6f}, {query_embedding[1]:.6f}, {query_embedding[2]:.6f}, {query_embedding[3]:.6f}, {query_embedding[4]:.6f}, ... {query_embedding[-1]:.6f}]")

# === Step 2: Perform k-NN vector similarity search ===
print(f"Searching vector index for top {TOP_K} results...\n")
search_body = {
    "size": TOP_K,
    "_source": True,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_embedding,
                "k": TOP_K
            }
        }
    }
}

response = os_client.search(index=INDEX_NAME, body=search_body)

# === Step 3: Display results with similarity scores ===
print(f"Top {TOP_K} results:\n")
print("-" * 70)
for rank, hit in enumerate(response["hits"]["hits"], start=1):
    score = hit["_score"]
    content = hit["_source"]["content"]
    emb = hit["_source"].get("embedding", [])
    print(f"  Rank {rank} | Similarity Score: {score:.4f}")
    print(f"  Chunk: {content[:100]}...")
    if emb:
        print(f"  Embedding: [{emb[0]:.6f}, {emb[1]:.6f}, {emb[2]:.6f}, {emb[3]:.6f}, {emb[4]:.6f}, ... {emb[-1]:.6f}]")
    print("-" * 70)

print("\n--- How similarity scores determine relevance ---")
print("  - Higher scores = stronger semantic match to your query")
print("  - Results are ranked by descending similarity score")
print("  - Semantic search matches meaning, not keywords")
