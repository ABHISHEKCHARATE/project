# Custom Chatbot using Langchain and Flask

## Overview
This project is a chatbot that extracts technical course data from [Brainlox](https://brainlox.com/courses/category/technical) using Langchain, processes it with Hugging Face embeddings, and serves responses via a Flask API.

## Features
- Scrapes course data from the web
- Splits text for better processing
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores embeddings in FAISS for quick retrieval
- Flask API to handle queries and return results

## Installation
Ensure you have Python installed, then install dependencies:
```sh
pip install flask langchain faiss-cpu sentence-transformers
```

## Usage
### Run the Flask app:
```sh
python app.py
```
### Access the home page:
Open your browser and visit:
```
http://localhost:5000/
```
### Send a query:
```sh
curl -X POST "http://localhost:5000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What courses are available?"}'
```
### Example Response:
```json
{
  "response": [
    {
      "content": "Course 1: Python Basics...",
      "source": "https://brainlox.com/courses/category/technical"
    },
    {
      "content": "Course 2: Data Science...",
      "source": "https://brainlox.com/courses/category/technical"
    }
  ]
}
```


