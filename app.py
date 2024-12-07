import os
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

index_name = "stocks"
namespace = "stock-descriptions"

hf_embeddings = HuggingFaceEmbeddings()

pinecone_index = pc.Index(index_name)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(page_title="Codebase RAG")

st.title("Financial Analysis LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_huggingface_embeddings(
    text, model_name="sentence-transformers/all-mpnet-base-v2"
):
    model = SentenceTransformer(model_name)
    return model.encode(text)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Enter your prompt here..."):
    raw_query_embedding = get_huggingface_embeddings(prompt)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace=namespace,
    )

    contexts = [item["metadata"]["text"] for item in top_matches["matches"]]

    augmented_query = (
        "<CONTEXT>\n"
        + "\n\n-------\n\n".join(contexts[:10])
        + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n"
        + prompt
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = "You are an expert at providing answers about stocks. Please answer my question provided."

    chat_completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            *[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            {"role": "user", "content": augmented_query},
        ],
        stream=True,
    )

    with st.chat_message("assistant"):
        chat_responses_generator = generate_chat_responses(chat_completion)
        full_response = st.write_stream(chat_responses_generator)

    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response}
        )
