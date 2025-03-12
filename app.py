from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load and process data
def load_and_store_data():
    url = "https://brainlox.com/courses/category/technical"
    loader = WebBaseLoader(url)

    # Load documents from the web page
    documents = loader.load()

    # Split the content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings using Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in FAISS vector database
    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store

# Load data and store embeddings
vector_store = load_and_store_data()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Renders the frontend page

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Perform similarity search
    results = vector_store.similarity_search(user_query, k=3)

    # Format response
    response = [{"content": res.page_content, "source": res.metadata.get("source", "Unknown")} for res in results]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
