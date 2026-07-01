import json
import time
import sys
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Configuration
REGION = "us-east-1"
INDEX_NAME = "support-docs"
MODEL_ID = "amazon.titan-embed-text-v2:0"

# Get the collection endpoint from the command line argument
if len(sys.argv) < 2:
    print("Usage: python3 create_index.py <collection-endpoint>")
    print("Example: python3 create_index.py https://abc123.us-east-1.aoss.amazonaws.com")
    sys.exit(1)

endpoint = sys.argv[1]
host = endpoint.replace("https://", "")
print(f"Collection endpoint: {endpoint}\n")

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

# --- Knowledge base documents (chunked support content) ---
DOCUMENTS = [
    "To return an item, log in to your account, go to Order History, select the item, "
    "and click Request Return. Print the prepaid shipping label and drop the package "
    "at any authorized carrier location within 30 days of delivery.",

    "Our standard shipping takes 5-7 business days. Expedited shipping delivers within "
    "2-3 business days for an additional fee. Free standard shipping is available on "
    "orders over $50.",

    "If your payment was declined, verify your card details and billing address match "
    "your bank records. You can update payment methods in Account Settings under "
    "Payment Options. Contact your bank if the issue persists.",

    "To cancel a subscription, navigate to Account Settings, select Subscriptions, "
    "choose the plan you want to cancel, and click Cancel Subscription. You will "
    "retain access until the end of your current billing period.",

    "Our warranty covers manufacturing defects for 12 months from the date of purchase. "
    "To file a claim, contact support with your order number and a description of the "
    "defect. Replacement or repair will be provided at no cost.",

    "Password reset can be done from the login page by clicking Forgot Password. Enter "
    "the email address associated with your account. A reset link will be sent within "
    "5 minutes. The link expires after 24 hours.",

    "Refunds are processed within 5-10 business days after we receive the returned item. "
    "The refund will appear on the original payment method. Partial refunds may apply "
    "for items returned in used condition.",

    "To track your order, go to Order History and click the tracking number link. "
    "You will be redirected to the carrier's website showing real-time delivery status "
    "and estimated arrival date.",
]


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


# === Step 1: Create the vector index ===
print("Creating vector index...")
index_body = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "space_type": "cosinesimil"
            }
        }
    }
}

try:
    os_client.indices.delete(index=INDEX_NAME)
    print("  Deleted existing index.")
    time.sleep(5)
except Exception:
    pass

os_client.indices.create(index=INDEX_NAME, body=index_body)
print("  Vector index created with knn_vector field (1024 dimensions, cosine similarity).\n")

# Wait for index to be ready for writes
time.sleep(10)

# === Step 2: Embed and index all documents ===
print("Embedding and indexing documents into OpenSearch...")
for i, doc in enumerate(DOCUMENTS):
    embedding = get_embedding(doc)
    os_client.index(
        index=INDEX_NAME,
        id=str(i),
        body={
            "content": doc,
            "embedding": embedding
        }
    )
    print(f"  Indexed document {i + 1}/{len(DOCUMENTS)}")

# === Step 3: Wait for all documents to become searchable ===
print("\nWaiting for all documents to become searchable...")
for attempt in range(30):  # Wait up to 5 minutes
    time.sleep(10)
    count_response = os_client.count(index=INDEX_NAME)
    doc_count = count_response["count"]
    print(f"  [{(attempt+1)*10}s] Documents searchable: {doc_count}/{len(DOCUMENTS)}")
    if doc_count >= len(DOCUMENTS):
        print("  All documents are searchable!")
        break
else:
    print(f"  Timed out. Only {doc_count} documents searchable.")

print("\nIndex created and documents embedded. You can now run:")
print(f"  python3 search.py {endpoint}")
