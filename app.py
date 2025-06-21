import streamlit as st
import os
import re
import dateparser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# Streamlit page config
st.set_page_config(
    page_title=' Task ChatBot ',
    page_icon=':robot_face:',
    layout='centered',
    initial_sidebar_state='auto'
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# Prompt template for Document QA
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided text only.
Provide the most accurate responses based on the question.
If the answer cannot be found in the context, respond that the information is not found in the provided documents.

<context>
{context}
<context>
Questions: {input}
""")

description = '''
A chatbot designed to answer questions directly from your uploaded documents. 
Using powerful language models and embeddings, it extracts answers from PDFs, research papers, reports, and more.
'''

def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def vector_embeddings(file):
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        st.session_state.docs = []
        st.session_state.final_documents = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        try:
            docs = loader.load()
            if not docs or len(docs) == 0:
                raise ValueError("The PDF has no readable text content.")
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

        final_documents = st.session_state.text_splitter.split_documents(docs)

        if not final_documents:
            raise ValueError("Document splitting produced no usable content.")

        st.session_state.docs.extend(docs)
        st.session_state.final_documents.extend(final_documents)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

        st.success("✅ Document uploaded and processed successfully.")
        st.info(f"Loaded {len(docs)} page(s) and split into {len(final_documents)} chunk(s).")

    except Exception as e:
        st.error(f"❌ Error during document processing: {e}")

# Validators for contact form and appointment form - return True/False
def validate_email(email):
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return bool(re.match(pattern, email))

def validate_phone(phone):
    pattern = r"^\+?[\d\s\-]{7,15}$"
    return bool(re.match(pattern, phone))

# Initialize session states for forms
if 'show_contact_form' not in st.session_state:
    st.session_state.show_contact_form = False
if 'show_appointment_form' not in st.session_state:
    st.session_state.show_appointment_form = False
if 'appointment_date' not in st.session_state:
    st.session_state.appointment_date = None

# Main UI
st.title('Task ChatBot')
st.sidebar.title('Upload Document')
st.sidebar.write(description)

file = st.sidebar.file_uploader('Upload your PDF document', type=['pdf'])
if file:
    vector_embeddings(file)

if st.sidebar.button('Refresh'):
    clear_session_state()
    st.experimental_rerun()

chat_mode = st.radio("Select Chat Mode:", ["Document QA", "General QA"])

user_msg = st.chat_input("Ask your question here...")
if user_msg:
    st.chat_message("User").write(user_msg)

    user_msg_lower = user_msg.lower()

    # Detect appointment booking intent
    if ('book appointment' in user_msg_lower or 'appointment' in user_msg_lower):
        parsed_date = dateparser.parse(user_msg, settings={'PREFER_DATES_FROM': 'future'})
        if parsed_date:
            st.session_state.appointment_date = parsed_date.date()
        else:
            st.session_state.appointment_date = None
        st.session_state.show_appointment_form = True
        st.session_state.show_contact_form = False
    elif 'call me' in user_msg_lower:
        st.session_state.show_contact_form = True
        st.session_state.show_appointment_form = False
    else:
        st.session_state.show_contact_form = False
        st.session_state.show_appointment_form = False

    if st.session_state.show_contact_form:
        st.markdown("### Please provide your contact details to get a call back")

        with st.form("contact_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone number (with country code)")
            email = st.text_input("Email")

            submitted = st.form_submit_button("Submit")

            if submitted:
                if not name.strip():
                    st.error("Please enter your name.")
                elif not validate_phone(phone):
                    st.error("Please enter a valid phone number.")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    st.success("Thank you! We will contact you soon.")
                    st.write(f"**Contact info submitted:** {name}, {phone}, {email}")
                    st.session_state.show_contact_form = False

    elif st.session_state.show_appointment_form:
        st.markdown("### Book an Appointment")

        default_date = st.session_state.appointment_date or None
        appointment_date = st.date_input("Select appointment date", value=default_date)

        with st.form("appointment_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone number (with country code)")
            email = st.text_input("Email")

            submitted = st.form_submit_button("Book Appointment")

            if submitted:
                if not name.strip():
                    st.error("Please enter your name.")
                elif not validate_phone(phone):
                    st.error("Please enter a valid phone number.")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    date_str = appointment_date.strftime("%Y-%m-%d") if appointment_date else "Unknown date"
                    st.success(f"Appointment booked for {date_str}!")
                    st.write(f"**Details:** {name}, {phone}, {email}")
                    st.session_state.show_appointment_form = False

    else:
        if chat_mode == "Document QA":
            if 'vectors' not in st.session_state:
                st.chat_message("Assistant").write("Please upload a document first.")
            else:
                try:
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    response = retrieval_chain.invoke({'input': user_msg})
                    answer = response.get("answer", "No answer returned from model.")
                    st.chat_message("Assistant").write(answer)
                except Exception as e:
                    st.chat_message("Assistant").write(f"Error while answering: {e}")

        elif chat_mode == "General QA":
            try:
                response = llm.invoke(user_msg)
                answer = getattr(response, "content", str(response))
                st.chat_message("Assistant").write(answer)
            except Exception as e:
                st.chat_message("Assistant").write(f"Error in general chat: {e}")

else:
    st.chat_message("Assistant").write("👋 Please type a question or say 'Call me' / 'Book appointment...' to provide contact info or book an appointment.")
