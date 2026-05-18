# AI Image Search Engine

An AI-powered semantic image search engine built using Flask, Qdrant, and CLIP embeddings.

## Features

- Upload images with tags
- Semantic text-to-image search
- AI-powered image embeddings
- Vector similarity search using Qdrant
- Modern frontend UI
- Flask backend integration

---

## Tech Stack

- Python
- Flask
- Qdrant Vector Database
- Sentence Transformers
- Jina CLIP v2
- HTML / CSS

---

## Project Architecture

AI-Image-Search-Engine
│
├── images
├── image_store
├── templates
│ └── index.html
├── requirements.txt
├── .gitignore
└── main.py

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Coderkrishn/AI-Image-Search-Engine.git
```

Move into project folder:

```bash
cd AI-Image-Search-Engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## How It Works

1. User uploads images
2. AI converts images into embeddings
3. Embeddings stored in Qdrant vector DB
4. User enters text query
5. Query converted into embedding
6. Semantic similarity search performed
7. Matching images displayed

---

## Future Improvements

- Image-to-image search
- Authentication system
- Cloud deployment
- Search history
- AI-generated tags
- Better UI animations

---

## Author

Krishn Kumar
