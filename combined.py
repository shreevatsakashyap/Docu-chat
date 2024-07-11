import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import HuggingFaceHub
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
import os

# Set environment variable for HuggingFaceHub API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Qdrant clients
url_langchain = "https://838581f9-41df-47dc-a452-47ea8f1e1edc.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key_langchain = "7QBIL3BHx33BOqbHlsSIrn5in7kTX4OLzn5FAXQYPFP7Y1KFYbagpg"
qdrant_client_langchain = QdrantClient(url=url_langchain, api_key=api_key_langchain)

url_haystack = "https://dc9421df-f74d-40b4-bec3-60e253854901.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key_haystack = "J__T4uhGMc1tqUW2aIn-Ik9ay1f0NIzeixs1APrkdEWqHmZcdUbpnQ"
qdrant_client_haystack = QdrantClient(url=url_haystack, api_key=api_key_haystack)

# Initialize Qdrant vector stores
qdrant_langchain = Qdrant(
    client=qdrant_client_langchain,
    collection_name="langchain",
    embeddings=embeddings
)

qdrant_haystack = Qdrant(
    client=qdrant_client_haystack,
    collection_name="my_collections",
    embeddings=embeddings
)

# Initialize the LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.5, "max_length": 1024, "max_new_tokens": 1024}
)

# Define the prompt template
question_answering_prompt_1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in LangChain, framework for building applications with large language models. Your task is to answer questions with the utmost accuracy, providing clear and detailed explanations. When relevant, include code sequences to illustrate your points and guide users effectively. Your responses should be informative, concise, and geared towards helping users understand and apply LangChain and Haystack concepts and techniques.\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
question_answering_prompt_2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in Haystack, framework for building applications with large language models. Your task is to answer questions with the utmost accuracy, providing clear and detailed explanations. When relevant, include code sequences to illustrate your points and guide users effectively. Your responses should be informative, concise, and geared towards helping users understand and apply LangChain and Haystack concepts and techniques.\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the document chain
document_chain_l = create_stuff_documents_chain(llm, question_answering_prompt_1)
document_chain_h = create_stuff_documents_chain(llm, question_answering_prompt_2)
# Streamlit app setup
st.set_page_config(page_title="LangChain & Haystack Chatbot", page_icon="🤖", layout="centered")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Layout for the Streamlit app
st.title('LangChain & Haystack Chatbot')

# Display chat history
for chat in st.session_state['chat_history']:
    if chat['role'] == 'user':
        st.markdown(f"**User:** {chat['content']}")
    else:
        st.markdown(f"**Assistant:** {chat['content']}")

# User input
query = st.text_input('Enter your query:', '')

# Documentation selection
doc_selection = st.selectbox('Select Documentation', ['LangChain', 'Haystack'])

if st.button('Send'):
    if query:
        # Add user message to chat history
        st.session_state['chat_history'].append({'role': 'user', 'content': query})

        # Initialize ephemeral chat history for the current interaction
        demo_ephemeral_chat_history = ChatMessageHistory()
        demo_ephemeral_chat_history.add_user_message(query)

        # Perform the similarity search based on user selection
        if doc_selection == 'LangChain':
            context = qdrant_langchain.similarity_search(query)
            response = document_chain_l.invoke(
            {
                "messages": demo_ephemeral_chat_history.messages,
                "context": context,
            }
        )
        else:
            context = qdrant_haystack.similarity_search(query)
            response = document_chain_h.invoke(
            {
                "messages": demo_ephemeral_chat_history.messages,
                "context": context,
            }
        )

        # Extract and display only the Assistant's message
        if 'Assistant:' in response:
            assistant_message = response.split('Assistant:')[-1].strip()
            st.session_state['chat_history'].append({'role': 'assistant', 'content': assistant_message})
            st.experimental_rerun()
        else:
            st.write("No valid response received.")
    else:
        st.write("Please enter a query.")
