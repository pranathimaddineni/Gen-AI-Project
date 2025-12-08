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
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    pass

# -----------------------------
# Top Bar
# -----------------------------
st.markdown('<div class="portfolio-banner">Pranathi Maddineni\'s Portfolio</div>', unsafe_allow_html=True)

# -----------------------------
# Projects Section
# -----------------------------
st.markdown("Hello, I’m Pranathi! I’m a passionate developer with experience in building dynamic e-commerce websites and a strong foundation in full-stack development (front-end focused). I enjoy creating projects that combine functionality with clean design and have added a few of my projects to GitHub to showcase my work. Currently, I’m exploring AI and machine learning, working on projects that leverage intelligent systems to solve real-world problems. I love learning, experimenting with new technologies, and bringing ideas to life through code.")
st.subheader("My Projects")
st.markdown("Click on a card to visit the project or view my resume/github profile")
st.markdown("Here are some of the e-commerce websites I've worked on, along with my resume and Github profile:")

projects = [
    {"name": "Jatango", "url": "https://www.jatango.com", "class": "card"},
    {"name": "Get Notifi", "url": "https://www.getnotifi.com", "class": "card"},
    {"name": "La-Excellence", "url": "https://laex.in/", "class": "card"},
    {"name": "Resume", "url": "https://drive.google.com/uc?export=download&id=1p5tNkCxvzCX72zjiitM0TGzGpdUqD1js", "class": "card"},
    {"name": "Github", "url": "HTTPS://GITHUB.COM/PRANATHIMADDINENI", "class": "card"}

]

# Create a flex container for cards
cards_html = '<div class="projects-container">'
for project in projects:
    cards_html += f'<a class="card {project["class"]}" href="{project["url"]}" target="_blank"><span>{project["name"]}</span></a>'
cards_html += '</div>'

st.markdown(cards_html, unsafe_allow_html=True)


# -----------------------------
# Chatbot Section
# -----------------------------

# -----------------------------
# Session state setup
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # Only store user & bot Q/A
if "question_asked" not in st.session_state:
    st.session_state.question_asked = False

# -----------------------------
# Top Bar
# -----------------------------

st.markdown("---") 
st.subheader("Chatbot")
st.markdown("A mini AI project demonstrating my practical skills in LangChain and OpenAI : a PDF-based chatbot that showcases my ability to build intelligent, interactive applications for real-world use.")
st.markdown("**Hello! Please upload your PDF file and start asking questions.**")  # static greeting

# -----------------------------
# PDF upload section
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.pdf_uploaded:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        t = page.extract_text()
        if t:
            text += t

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
    st.session_state.pdf_uploaded = True

    st.markdown("PDF uploaded successfully!")  # PDF uploaded message shown once

# -----------------------------
# Render previous messages (user Q & bot A)
# -----------------------------
for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.markdown(f"{msg['content']}")
#     else:
        st.markdown(f"{msg['content']}")

# -----------------------------
# Chat input
# -----------------------------
if st.session_state.pdf_uploaded and not st.session_state.question_asked:
    question = st.text_input("Ask a question:", key="ask_input")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.question_asked = True

        # Retrieve relevant chunks
        retriever = st.session_state.vector_store.as_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        # LLM response
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer using ONLY the provided PDF text."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]
        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.messages.append({"role": "bot", "content": answer})

     #    st.markdown(f"{question}")
        st.markdown(f"{answer}")