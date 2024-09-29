import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt_template = """
    Assume that the only information you have is the context given nothing else.
    The use of the contextual knowledge must be evident. Answer in an descriptive format.
    Answer the question in detail from the provided document, make sure to cover all the relevant points, there should be no additonal information, only reply based on document.\n
    Use on the contexts to formulate the response. The response must NOT contain anything that is not there in the context give.If the answer is not provided in the context just say, "answer is not available in the context".\n
    Dont provide the wrong answer.do not mention anything not in the document, the entire response must be crafted strictly from what is avaliabe in the context. Also dont add any extra observations or considerations on your own if not present in the context\n
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a memory buffer for conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Create the ConversationalRetrievalChain with the custom prompt
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,  # We can handle this through the prompt
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return conversation_chain

import re

def clean_and_truncate_text(text, word_limit=50):
    # Remove broken words (words split by spaces like "Spri ng Bo ot")
    cleaned_text = re.sub(r'(\w+)\s+(\w+)', lambda match: match.group(1) + match.group(2) if match.group(1).lower() in ['spring', 'boot', 'java', 'project'] else ' '.join(match.groups()), text)
    
    # Replace excessive spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Limit to the first 50 words and add ellipsis
    truncated_text = ' '.join(cleaned_text.split()[:word_limit]) + '...'
    
    return truncated_text

def handle_userinput(user_question):
    if st.session_state.conversation:
        # Get the top 3 most relevant documents (text chunks)
        top_k_docs = st.session_state.conversation.retriever.get_relevant_documents(user_question)[:3]

        # Generate the content of the context as a numbered and cleaned list
        context_paragraphs = ""
        for i, doc in enumerate(top_k_docs, 1):
            cleaned_text = clean_and_truncate_text(doc.page_content)
            context_paragraphs += f"{i}. {cleaned_text}\n\n"

        # Get the model's response
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Display chat history (user and bot), attach context dropdown to bot responses
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                # User message
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # Bot response
                bot_response = bot_template.replace(
                    "{{MSG}}", message.content)
                st.write(bot_response, unsafe_allow_html=True)

                # Show context as an expander (instead of <details>)
                with st.expander("Show relevant context"):
                    st.write(context_paragraphs)

    else:
        st.write("Conversation chain is not initialized. Please process your documents first.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with documents!", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store with Google embeddings
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain with Gemini model
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
