import os
import uuid

import torch
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from flask import Flask, request, render_template

# Flask App
app = Flask(__name__, static_folder='images', static_url_path='/images')

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# AI Model
model = SentenceTransformer(
    'jinaai/jina-clip-v2',
    trust_remote_code=True,
    truncate_dim=1024,
    device=device
)

# Qdrant Database
client = QdrantClient(path='image_store')

# Create Collection
if not client.collection_exists(collection_name='images'):

    client.create_collection(
        collection_name='images',

        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Upload Image Route
@app.route('/upload_image', methods=['POST'])
def upload_image():

    file = request.files['image_file']

    tags = request.form.get('image_tags', '')

    image_path = os.path.join('images', file.filename)

    # Save Image
    file.save(image_path)

    # Generate Embedding
    image_embedding = model.encode(
        image_path,
        normalize_embeddings=True,
        device=device
    )

    # Store in Qdrant
    client.upsert(
        collection_name='images',

        points=[
            PointStruct(
                id=str(uuid.uuid4()),

                vector=image_embedding.tolist(),

                payload={
                    'path': image_path,
                    'tags': tags
                }
            )
        ]
    )

    return render_template('index.html')


# Search Route
@app.route('/search_query', methods=['POST'])
def search_query():

    search_query = request.form['query']

    # Convert Text to Embedding
    embedded_query = model.encode(
        search_query,
        normalize_embeddings=True,
        device=device
    )

    # Tag Search
    tag_filter = Filter(
        must=[
            FieldCondition(
                key='tags',
                match=MatchValue(value=search_query)
            )
        ]
    )

    tag_results = client.query_points(
        collection_name='images',
        query=embedded_query,
        query_filter=tag_filter,
        limit=5
    ).points

    # Semantic Search
    results = client.query_points(
        collection_name='images',
        query=embedded_query,
        limit=5 - len(tag_results)
    ).points

    # Merge Results
    final_results = tag_results

    for r in results:

        if r not in final_results:
            final_results.append(r)

    # Send Results to HTML
    return render_template(
        'index.html',
        results=[r.payload['path'] for r in final_results]
    )


# Run App
if __name__ == '__main__':
   app.run(debug=False)
    # Run App
