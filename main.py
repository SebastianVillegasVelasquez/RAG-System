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
                                   llm,
                                   history):
    """
        Generate an AI response based on research documents.

        Performs semantic search in the vector database, builds a context
        from the retrieved documents, and queries a language model to
        generate an answer.

        Args:
            query (str): The researcher's question.
            collections (chromadb.Collection): ChromaDB collection instance.
            llm: Language model instance (e.g., ChatGroq).
            history: Conversation context from streamlit session

        Returns:
            str: AI-generated response based on the retrieved research context.
        """
    results = semantic_search_in_collection(query, collections=collections, top_k=3)
    context = "\n\n".join([
        f"From:\n{chunk}"
        for chunk in results["documents"]
    ])
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "history"],
        template="""
        You are an AI research assistant. 
        Answer the following question based strictly on the research findings provided. 
        Keep your answer clear, concise, and focused on the relevant information. 
        Do not invent information. If the context does not contain enough evidence to 
        answer the question, state that clearly.
        
        conversation so far:
        {history}
        
        Research Context:
        {context}
        
        Researcher's Question:
        {question}
        
        Answer:provide a comprehensive answer based on the research findings above.
        """
    )

    prompt = prompt_template.format(context=context, question=query, history=history)
    response = llm.invoke(prompt)

    return response.content


def show_chat_page(llm, collection: chromadb.Collection):
    """
           Display a Streamlit-based chat interface for interacting with documents.

           Renders a chat UI where the user can type messages, stores the conversation
           history in the session state, and generates responses using an LLM and
           semantic search over a ChromaDB collection.

           Args:
               llm: Language model instance used to generate responses.
               collection (chromadb.Collection): ChromaDB collection instance
                                                 containing the document embeddings.

           Workflow:
               1. Display chat title.
               2. Initialize chat history in the session state (if not present).
               3. Capture user input from chat box.
               4. Generate AI response using the LLM and document collection.
               5. Append both user and assistant messages to session history.
               6. Render the entire conversation in the chat interface.
       """
    st.title("How can I help you today?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = generate_answer_from_documents(
            query=user_input,
            collections=collection,
            llm=llm,
            history=st.session_state.messages
        )

        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == '__main__':
    collection = load_and_insert_researches()
    llm = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv('API_KEY'))
    show_chat_page(llm=llm, collection=collection)
