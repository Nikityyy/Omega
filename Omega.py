#pip install -qq langchain wget llama-index cohere llama-cpp-python pytesseract opencv-python-headless pytube youtube-transcript-api
#pip -q install streamlit pymupdf beautifulsoup4

import requests
from bs4 import BeautifulSoup
from docx import Document
from concurrent.futures import ThreadPoolExecutor
import os
from PIL import Image
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import pyperclip
import shutil
import wget
import numpy as np
import pytesseract
import streamlit as st
import tempfile
import fitz
import time
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

def bar_custom(current, total, width=80):
    print("Downloading %d%% [%d / %d] bytes" % (current / total * 100, current, total))

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}")
        wget.download(model_url, out=model_path)

def init_page() -> None:
    st.set_page_config(
        page_title="Omega",
        page_icon="Smile.png",
    )
    st.title("Omega")
    st.sidebar.title("Options")

def select_llm(model_path) -> LlamaCPP:
    print(f"Checking if model exists at {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure it is downloaded.")

    print("Initializing Omega...")
    return LlamaCPP(
        model_path=model_path,
        temperature=0.45,
        max_new_tokens=200,
        context_window=2000,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 15},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="Hello! My name is Omega, and I'm here to help you with any questions or tasks you may have. Please let me know how I can assist you further."
            )
        ]

def get_answer(llm, messages) -> str:
    response = llm.complete(messages)
    return response.text

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {e}")
        return None
        
def extract_text_from_docx(uploaded_file):
    try:
        # Load the DOCX document
        doc = Document(uploaded_file)

        # Extract text from each paragraph
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def perform_ocr(image):
    # Resize the image to reduce processing time
    resized_image = image.resize((800, 600))

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(resized_image)
    return text

def extract_text_from_pdf(uploaded_file):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save the PDF content to a temporary file within the temporary directory
        temp_file_path = os.path.join(temp_dir, "temp.pdf")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        # Open the temporary PDF file with fitz
        with fitz.open(temp_file_path) as pdf_document:
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
    finally:
        # Remove the temporary directory and its contents
        #shutil.rmtree(temp_dir)
        print("")
    
    return text

def get_session_state():
    if "show_example_message" not in st.session_state:
        st.session_state.show_example_message = True
    return st.session_state.show_example_message

def reload():
    st.markdown(
    """
    <script>
        window.location.reload(true);
    </script>
    """,
    unsafe_allow_html=True,
    )
    
def extract_subtitles_from_url(video_url):
    try:
        # Extract video ID from the URL
        video_id = YouTube(video_url).video_id

        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, 
        languages = [
            'af', 'ak', 'sq', 'am', 'ar', 'hy', 'as', 'ay', 'az', 'bn',
            'eu', 'be', 'bh', 'bs', 'bg', 'my', 'ca', 'ce', 'zh', 'zh',
            'co', 'hr', 'cs', 'da', 'dv', 'nl', 'en', 'eo', 'et', 'ee',
            'fi', 'fi', 'fr', 'gl', 'lg', 'ka', 'de', 'el', 'gn', 'gu',
            'ht', 'ha', 'ha', 'iw', 'hi', 'hm', 'hu', 'is', 'ig', 'id',
            'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'ki', 'ko', 'kr',
            'ku', 'ky', 'lo', 'la', 'lv', 'li', 'lt', 'lb', 'mk', 'mg',
            'ms', 'ml', 'mt', 'mā', 'mr', 'mn', 'ne', 'ns', 'no', 'ny',
            'or', 'om', 'ps', 'fa', 'pl', 'pt', 'pa', 'qu', 'ro', 'ru',
            'sa', 'sa', 'gd', 'sr', 'sn', 'sd', 'si', 'sk', 'sl', 'so',
            'st', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'tt', 'te', 'th',
            'ti', 'ts', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy',
            'fy', 'xh', 'yi', 'yo', 'zu'
        ])
        
        # Extract subtitles
        subtitles = [entry['text'] for entry in transcript]
        
        return subtitles
    except Exception as e:
        print(f"Error: {e}")
        return None

def set_session_state(value):
    st.session_state.show_example_message = value

def main():
    #model_url = "https://huggingface.co/TheBloke/TheBloke/llama2_7b_chat_uncensored-GGUF/resolve/main/llama2_7b_chat_uncensored.Q2_K.gguf"
    #model_filename = "llama2_7b_chat_uncensored.Q2_K.gguf"
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"
    model_filename = "llama-2-7b-chat.Q2_K.gguf"
    #model_url = "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf"
    #model_filename = "llama-2-13b-chat.Q5_K_M.gguf"
    model_path = os.path.join(os.getcwd(), model_filename)
    
    download_model(model_url, model_path)

    init_page()
    with st.spinner("Initializing Omega..."):
        llm = select_llm(model_path)
    init_messages()
    
    if user_input := st.chat_input("Search the World!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking ..."):
            answer = get_answer(llm, user_input)
            print(answer)
        st.session_state.messages.append(AIMessage(content=answer))
    
    # Add clickable example message below the title
    example_message = "How to center a div"
    show_example_message = get_session_state()
    if st.button(example_message, key="example_button") and show_example_message:
        st.session_state.messages.append(HumanMessage(content=example_message))
        with st.spinner("Thinking ..."):
            answer = get_answer(llm, example_message)
            set_session_state(False)
        st.session_state.messages.append(AIMessage(content=answer))
        set_session_state(False)

    # Add file uploader for both images and PDFs
    uploaded_file = st.file_uploader("Upload File", type=["docx", "png", "pdf", "jpg"])
    if uploaded_file is not None:
        # Check if the file is a PDF
        if uploaded_file.type == 'application/pdf':
            # Extract text from the PDF
            text_from_pdf = extract_text_from_pdf(uploaded_file)
            
            # Display the extracted text
            st.write(f"Copied extracted Text from PDF")
            
            pyperclip.copy(text_from_pdf)
            
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # Extract text from the DOCX file
            text_from_docx = extract_text_from_docx(uploaded_file)

            if text_from_docx:
                # Display the extracted text
                st.write(f"Copied extracted Text from DOCX")
                pyperclip.copy(text_from_docx)
                
            else:
                st.warning("Failed to extract text from the DOCX file. Please check the file format and try again.")
        
        else:
            # Perform OCR on the uploaded image
            image = Image.open(uploaded_file)
            text_from_image = perform_ocr(image)
            
            # Display the extracted text
            st.write(f"Copied extracted Text from Image")
            
            pyperclip.copy(text_from_image)
            
        reload()
            
    url_input = st.text_input("Do you want to use a website as context? Enter an URL:")

    if url_input:
        if 'http' not in url_input:
            if '://' not in url_input:
                url_input = 'https://' + url_input

        # Fetch text content from the URL
        url_content = fetch_text_from_url(url_input)

        if url_content:
            # Display the fetched content
            st.write(f"Copied text from {url_input}")
            
            pyperclip.copy(url_content)
        else:
            st.warning("Failed to fetch content from the URL. Please check the URL and try again.")
            
    yt_input = st.text_input("Do you want to summarize a YT Video? Enter an URL:")

    if yt_input:

        # Fetch text content from the URL
        yt_content = extract_subtitles_from_url(yt_input)

        if yt_content:
            # Display the fetched content
            #st.write(f"Copied text from {yt_input}")
            
            yt_content_str = ' '.join(yt_content)
            
            summarize_yt = f'Summarize the following text "{yt_content_str}"'
            
            st.session_state.messages.append(HumanMessage(content=summarize_yt))
            with st.spinner("Thinking ..."):
                answer = get_answer(llm, summarize_yt)
                print(answer)
            st.session_state.messages.append(AIMessage(content=answer))
        else:
            st.warning("Failed to fetch content from the URL. Please check the URL and try again.")

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
          with st.chat_message("Omega", avatar="Smile.png"):
            st.markdown(message.content)
        elif isinstance(message, HumanMessage):
          with st.chat_message("user"):
            st.markdown(message.content)

if __name__ == "__main__":
    main()
    
#streamlit run Omega.py & npx localtunnel --port 8501
