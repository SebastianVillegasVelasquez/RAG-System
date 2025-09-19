import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from modules import generate_embeddings, load_and_insert_researches

load_dotenv()


def semantic_search_in_collection(
        query: str,
        collections: chromadb.Collection,
        top_k=5
):
    """
        Perform semantic search in the vector database.

        Encodes the input query into an embedding vector and retrieves the
        most relevant results from the ChromaDB collection.

        Args:
            query (str): The search query.
            collections (chromadb.Collection): ChromaDB collection instance.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            dict: Query results containing documents, metadata, and distances.
        """
    embedded_query = generate_embeddings([{"content": query}])[0]

    return collections.query(
        query_embeddings=embedded_query,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )


def generate_answer_from_documents(query: str,
                                   collections: chromadb.Collection,
                                   llm):
    """
        Generate an AI response based on research documents.

        Performs semantic search in the vector database, builds a context
        from the retrieved documents, and queries a language model to
        generate an answer.

        Args:
            query (str): The researcher's question.
            collections (chromadb.Collection): ChromaDB collection instance.
            llm: Language model instance (e.g., ChatGroq).

        Returns:
            str: AI-generated response based on the retrieved research context.
        """
    results = semantic_search_in_collection(query, collections=collections, top_k=3)
    context = "\n\n".join([
        f"From:\n{chunk}"
        for chunk in results["documents"]
    ])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI research assistant. 
        Answer the following question based strictly on the research findings provided. 
        Keep your answer clear, concise, and focused on the relevant information. 
        Do not invent information. If the context does not contain enough evidence to 
        answer the question, state that clearly.
        
        Research Context:
        {context}
        
        Researcher's Question:
        {question}
        
        Answer:provide a comprehensive answer based on the research findings above.
        """
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    return response.content


def show_chat_page(llm, collection: chromadb.Collection):
    st.title("How can I help you today?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = generate_answer_from_documents(
            query=user_input,
            collections=collection,
            llm=llm
        )

        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == '__main__':
    collection = load_and_insert_researches()
    llm = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv('API_KEY'))
    show_chat_page(llm=llm, collection=collection)
