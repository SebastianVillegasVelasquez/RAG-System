import os

import chromadb
import torch
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from src.paths import DATABASE_DIR, DOCUMENTS_PATH


@st.cache_resource
def initialize_chromadb_collection(name: str = 'cancer_collection'):
    """
        Initialize and retrieve a ChromaDB collection.

        This function creates a persistent ChromaDB client at the predefined
        `DATABASE_DIR` path and either retrieves or creates a collection
        with the given name. It uses cosine similarity as the distance metric
        for embedding comparison.

        Args:
            name (str, optional): Name of the collection. Defaults to 'cancer_collection'.

        Returns:
            chromadb.Collection: A ChromaDB collection instance ready for document insertion and querying.
        """
    client = chromadb.PersistentClient(path=DATABASE_DIR)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def load_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                         device: str = 'cpu'):
    """
        Load a HuggingFace embedding model.

        This function initializes a HuggingFace embedding model for text
        representation. It can be configured to run on CPU, CUDA, or MPS
        depending on the available hardware.

        Args:
            model_name (str, optional): HuggingFace model name. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            device (str, optional): Device to load the model ('cpu', 'cuda', or 'mps'). Defaults to 'cpu'.

        Returns:
            HuggingFaceEmbeddings: An embedding model instance.
        """
    return HuggingFaceEmbeddings(model_name=model_name,
                                 model_kwargs={'device': device})


def load_text_publications(documents_dir: str) -> list[str]:
    """
        Load research publications from text files.

        Reads `.txt` files from the given directory and extracts their content
        into structured dictionaries containing the text and file title.

        Args:
            documents_dir (str): Path to the directory containing text documents.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - 'content' (str): The text of the document.
                - 'title' (str): The file name used as the title.

        Raises:
            Exception: If a document cannot be read or parsed.
        """
    documents = []
    for document in os.listdir(documents_dir):
        file_name = os.path.join(documents_dir, document)
        if file_name.endswith('.txt'):
            try:
                loader = TextLoader(file_path=file_name, encoding='utf-8')
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append({
                        "content": doc.page_content,
                        "title": os.path.basename(file_name)
                    })
                print(f'document successfully loaded: {file_name}')
            except Exception as e:
                print(f'Failed to load the {file_name} document\n'
                      f'exception: {e}')
    return documents


def split_document_into_chunks(
        document: dict,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
) -> list:
    """
        Split a document into overlapping text chunks.

        Uses `RecursiveCharacterTextSplitter` to break down long documents
        into smaller chunks suitable for embedding and retrieval.
        Metadata such as title and chunk ID are added for traceability.

        Args:
            document (dict): A dictionary containing:
                - 'content' (str): The full text of the document.
                - 'title' (str): The title or file name of the document.
            chunk_size (int, optional): Maximum size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

        Returns:
            list[dict]: A list of chunk dictionaries containing:
                - 'content' (str): The chunked text.
                - 'title' (str): Original document title.
                - 'chunk_id' (str): Unique chunk identifier.
        """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks_from_publication = text_splitter.split_text(document["content"])
    title = str(document['title']).replace(".txt", "")
    return [
        {
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}"
        }
        for i, chunk in enumerate(chunks_from_publication)
    ]


def generate_embeddings(
        chunks_document: list[dict]
) -> list[list[float]]:
    """
        Generate embeddings for a list of text chunks.

        Selects the best available device (CUDA, MPS, or CPU), loads the
        embedding model, and encodes each text chunk into a vector
        representation.

        Args:
            chunks_document (list[dict]): List of dictionaries containing:
                - 'content' (str): Text to embed.

        Returns:
            list[list[float]]: A list of embedding vectors.
        """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = load_embedding_model(device=device)
    texts = [text["content"] for text in chunks_document]
    embeddings = model.embed_documents(texts)
    print(f"[debug] embedded {len(embeddings)} chunks; first vector length: {len(embeddings[0]) if embeddings else 0}")
    return embeddings


def insert_documents_into_collection(collection: chromadb.Collection,
                                     documents: list):
    """
    Insert documents into a ChromaDB collection.

    This function chunks each document, generates embeddings, and stores
    them in the given ChromaDB collection along with their metadata.

    Args:
        collection (chromadb.Collection): ChromaDB collection instance.
        documents (list[dict]): List of documents where each document contains:
            - 'content' (str): Full document text.
            - 'title' (str): Document title.

    Raises:
        Exception: If the insertion process fails.
    """
    next_id = collection.count()
    for document in documents:
        chunked_publication = split_document_into_chunks(document=document)
        embeddings = generate_embeddings(chunks_document=chunked_publication)
        ids = [f"document_{next_id + i}" for i in range(len(chunked_publication))]
        documents_texts = [c["content"] for c in chunked_publication]
        metadatas = [{"title": c["title"], "chunk_id": c["chunk_id"]} for c in chunked_publication]

        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_texts,
                metadatas=metadatas
            )
            next_id += len(chunked_publication)
            print(f"Inserted {len(chunked_publication)} chunks (ids {ids[0]} ...)")
        except Exception as e:
            print(f"Error while inserting publication '{document.get('title')}' : {e}")

@st.cache_resource
def load_and_insert_researches():
    collection = initialize_chromadb_collection()
    documents = load_text_publications(documents_dir=DOCUMENTS_PATH)
    insert_documents_into_collection(collection=collection, documents=documents)
    return collection