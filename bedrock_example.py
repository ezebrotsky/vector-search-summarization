import boto3
import pymongo
import json
import os

from botocore.exceptions import ClientError
from bson.objectid import ObjectId
from dotenv import load_dotenv
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# setup Mongo client
mongo_client = pymongo.MongoClient(MONGO_URI)

# define db and collection
db = mongo_client.sample_responses
collection = db.unstructured

def connect_to_bedrock():
    # Initialize a session using Boto3
    session = boto3.Session()

    # Create a Bedrock client
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # Create a Bedrock client
    bedrock_client = session.client('bedrock-runtime')
    
    return bedrock_client

def get_bedrock_embedding(input_text: str):
    try:
        client = connect_to_bedrock()

        # Create the request for the model.
        native_request = {"inputText": input_text}

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        # Invoke the model with the request.
        response = client.invoke_model(modelId='amazon.titan-embed-text-v1', body=request)

        # Decode the model's native response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the generated embedding and the input text token count.
        embedding = model_response["embedding"]
        input_token_count = model_response["inputTextTokenCount"]

        print(f"Number of input tokens: {input_token_count}")

        return embedding

    except Exception as e:
        print(f"An error occurred: {e}")

def generate_database_embedding():
    try:
        for doc in collection.find({'response':{"$exists": True, '$ne': None}, 'embedding': {'$size': 0}}):
            doc['embedding'] = get_bedrock_embedding(doc['response'])
            collection.replace_one({'_id': doc['_id']}, doc)
            print(f"Document {str(doc['_id'])} updated")


        
    except Exception as e:
        print(f"An error occurred: {e}")

def process_document(doc):
    # Process the document
    doc['embedding'] = get_bedrock_embedding(doc['response'])
    collection.replace_one({'_id': doc['_id']}, doc)
    print(f"Document {str(doc['_id'])} updated")
    # Your processing logic here

def test_conversation(responses: list):
    try:
        results = '\n'.join([str(elem) for elem in responses])

        client = connect_to_bedrock()

        # Set the model ID, e.g., Titan Text Premier.
        model_id = "amazon.titan-text-premier-v1:0"

        # Define the prompt for the model.
        prompt = f"""
        The following is the result of a search in Google:
        
        {results}

        Summarize the above list of responses.
        """

        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        try:
            # Invoke the model with the request.
            response = client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
            )

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
    except Exception as e:
        print(f"An error occurred: {e}")

def query(lineItemId: str, prompt: str):
    try:
        # Create query
        embedding = get_bedrock_embedding(prompt)

        results = collection.aggregate(
            [
                {
                    '$vectorSearch': {
                        'index': 'response_vector',
                        'path': 'embedding',
                        'filter': {'lineItemId': {'$eq': ObjectId(lineItemId)}},
                        'queryVector': embedding, 
                        'numCandidates': 1000, 
                        'limit': 50
                    }
                }, {
                    '$project': {
                        '_id': 0, 
                        'response': 1, 
                        'score': {
                            '$meta': 'vectorSearchScore'
                        }
                    }
                }
            ]
        )

        test_conversation(results)
    except Exception as e:
        print(f"An error occurred: {e}")

def create_vector_search_index():
    # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
    # Create your index model, then create the search index

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "numDimensions": 1536,
                    "path": "embedding",
                    "similarity": "cosine",
                    "type": "vector"
                },
                {
                    "path": "lineItemId",
                    "type": "filter"
                }
            ]

        },
        name="response_vector",
        type="vectorSearch"
    )

    result = collection.create_search_index(model=search_index_model)

    print(result)


query('66a7b8df5c4280aedf5d5951', 'Retrieve results where the description is about someone who has died that includes a date')