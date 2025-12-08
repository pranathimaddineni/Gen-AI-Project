import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Load external CSS
# -----------------------------
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

# -----------------------------
# Top Banner
# -----------------------------
st.markdown(
    '<div class="portfolio-banner">Pranathi Maddineni\'s Portfolio</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Portfolio Introduction
# -----------------------------
st.markdown("""
Hello, I’m Pranathi! I’m a passionate developer with experience in building dynamic e-commerce websites and a strong foundation in full-stack development (front-end focused). I enjoy creating projects that combine functionality with clean design, and I've added a few of my projects to GitHub to showcase my work.

Currently, I’m exploring AI and machine learning, working on projects that leverage intelligent systems to solve real-world problems.
""")

st.subheader("My Projects")
st.markdown("Click on a card to visit the project or view my resume/github profile")

projects = [
    {"name": "Jatango", "url": "https://www.jatango.com", "class": "card"},
    {"name": "Get Notifi", "url": "https://www.getnotifi.com", "class": "card"},
    {"name": "La-Excellence", "url": "https://laex.in/", "class": "card"},
    {"name": "Resume", "url": "https://drive.google.com/uc?export=download&id=1p5tNkCxvzCX72zjiitM0TGzGpdUqD1js", "class": "card"},
    {"name": "Github", "url": "https://github.com/pranathimaddineni", "class": "card"}
]

cards = '<div class="projects-container">'
for p in projects:
    cards += f'<a class="card" href="{p["url"]}" target="_blank"><span>{p["name"]}</span></a>'
cards += '</div>'

st.markdown(cards, unsafe_allow_html=True)

# -----------------------------
# Chatbot Section
# -----------------------------
st.markdown("---")
st.subheader("Chatbot")
st.markdown("Upload a PDF and ask any questions about its content!")

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# PDF Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.pdf_uploaded:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
    st.session_state.pdf_uploaded = True

    st.success("PDF uploaded successfully!")

# -----------------------------
# Chat History Container
# -----------------------------
chat_area = st.container()

with chat_area:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")


if st.session_state.pdf_uploaded:
    question = st.text_input("Ask any question you want to know from the PDF:", key="chat_input")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        # Retrieve context
        retriever = st.session_state.vector_store.as_retriever()
        try:
            docs = retriever.get_relevant_documents(question)
        except:
            docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0
        )

        msgs = [
            SystemMessage(content="Answer ONLY using the provided PDF content."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]

        # AI response
        try:
            response = llm.invoke(msgs)
            answer = response.content
        except Exception as e:
            answer = f"Error: {e}"

        # Save and re-render
        st.session_state.messages.append({"role": "bot", "content": answer})

        # Re-render whole chat AFTER append
        chat_area.empty()
        with chat_area:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Bot:** {msg['content']}")

else:
    st.info("Upload a PDF to start chatting.")
