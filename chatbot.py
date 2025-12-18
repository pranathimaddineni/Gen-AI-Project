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
Hello, I’m Pranathi! I’m a passionate developer with experience in building dynamic e-commerce and internal web applications and a strong foundation in full-stack development (front-end focused). I enjoy creating projects that combine functionality with clean design, and I've added a few of my projects to GitHub to showcase my work.

Currently, I’m also exploring AI and machine learning, working on projects that leverage intelligent systems to solve real-world problems.
""")

st.subheader("My Projects")
st.markdown("Click on a card to visit the project or view my resume/github profile")

projects = [
    {"name": "Jatango", "url": "https://www.jatango.com"},
    {"name": "Get Notifi", "url": "https://www.getnotifi.com"},
    {"name": "La-Excellence", "url": "https://laex.in/"},
    {"name": "Resume", "url": "https://drive.google.com/uc?export=download&id=1p5tNkCxvzCX72zjiitM0TGzGpdUqD1js"},
    {"name": "Github", "url": "https://github.com/pranathimaddineni"}
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
st.markdown("A mini AI project demonstrating my practical skills in LangChain and OpenAI : a chatbot that showcases my ability to build intelligent, interactive applications for real-world use.")
st.markdown("Upload a PDF and ask any questions about its content!")

# -----------------------------
# Session State Initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

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

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
    st.session_state.pdf_uploaded = True
    st.success("PDF uploaded successfully!")

# -----------------------------
# Chat History Rendering
# -----------------------------
chat_area = st.container()

def render_chat():
    # chat_area.empty()
    with chat_area:
        for msg in st.session_state.messages:
            role = "**You:**" if msg["role"] == "user" else "**Bot:**"
            st.markdown(f"{role} {msg['content']}")

# -----------------------------
# Chat Input & Bot Response
# -----------------------------
def handle_question():
    question = st.session_state.chat_input.strip()
    if not question:
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Check if already answered
    if question in st.session_state.qa_history:
        answer = st.session_state.qa_history[question]
    else:
        # Retrieve relevant chunks
        retriever = st.session_state.vector_store.as_retriever()
        try:
            docs = retriever.get_relevant_documents(question)
        except:
            docs = []

        context = "\n\n".join([d.page_content for d in docs])

        # Prepare prompt
        if context.strip():
            prompt = f"Answer the question using ONLY the following context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = f"Answer the question:\n{question}"

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0
        )

        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]

        try:
            response = llm.invoke(msgs)
            answer = response.content.strip()
        except Exception as e:
            answer = f"Error: {e}"

        # Cache answer
        st.session_state.qa_history[question] = answer

    # Append bot message
    st.session_state.messages.append({"role": "bot", "content": answer})

    # Clear input
    st.session_state.chat_input = ""

# -----------------------------
# Show chat input
# -----------------------------
# -----------------------------
# Chat Input 
# -----------------------------
if st.session_state.pdf_uploaded:
    user_input = st.chat_input("Ask a question about the PDF")
    
    if user_input:
        # append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k":4})
        try:
            docs = retriever.invoke(user_input)
        except Exception:
            docs = []

        context = "\n\n".join([d.page_content for d in docs])

        SYSTEM_PROMPT = """
You are an AI assistant analyzing a RESUME uploaded by the user.

Answer based ONLY on the uploaded resume.
If info is missing, say "The resume does not mention this."
"""
        human_prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt)
        ])
        answer = response.content.strip()

        st.session_state.messages.append({"role":"assistant","content":answer})

        # re-render chat
        for msg in st.session_state.messages:
            role = "**You:**" if msg["role"] == "user" else "**Bot:**"
            st.markdown(f"{role} {msg['content']}")
else:
    st.info("Upload a PDF to start chatting.")

# # Render chat messages
# render_chat()
