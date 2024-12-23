import os
import gradio as gr
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
import pytesseract  # For image text extraction
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set up Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it up.")
client = Groq(api_key=api_key)
llm = ChatGroq(model_name='llama3-70b-8192')

# Initialize global variables
retriever = None
qa_chain = None

def process_and_load_document(file):
    """Loads, splits, and embeds the uploaded document."""
    global retriever, qa_chain
    file_extension = file.name.split('.')[-1].lower()

    # Handle document types
    if file_extension == "pdf":
        loader = PyPDFLoader(file.name)
        data = loader.load()
    elif file_extension == "docx":
        loader = Docx2txtLoader(file.name)
        data = loader.load()
    elif file_extension == "txt":
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
        data = [{"text": text}]
    elif file_extension == "csv":
        df = pd.read_csv(file.name)
        text = df.to_string()
        data = [{"text": text}]
    elif file_extension in ["png", "jpg", "jpeg"]:
        # Extract text from image using Tesseract OCR
        image = Image.open(file.name)
        text = pytesseract.image_to_string(image)
        data = [{"text": text}]
    else:
        return "Unsupported file type. Please upload a PDF, DOCX, TXT, CSV, or image file."

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = text_splitter.split_documents(data)

    # Create embeddings and vector store
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_documents(text, embeddings)

    # Setup retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Setup QA chain
    prompt_template = '''
    You are a helpful assistant. Greet the user before answering.

    {context}
    {question}
    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           chain_type_kwargs={"prompt": prompt},
                                           return_source_documents=True)
    return "📄 Document uploaded and processed successfully!"

def ask_question(query):
    """Handles user queries by leveraging the QA chain."""
    if not qa_chain:
        return "⚠️ Please upload a document first."
    result = qa_chain.invoke({"query": query})  # Use the new invoke method
    return f"💬 **Answer:** {result['result']}"

# Gradio Interface
# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat with Documents using Groq AI</h1>")
    gr.Markdown(
        "<p style='text-align: center; color: #fff;'>Upload a document (PDF, DOCX, TXT, CSV, or Image) and ask questions to interact with its content!</p>"
    )
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="📁 Upload Document", type="filepath")
            upload_button = gr.Button("Upload and Process", elem_id="upload_button")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
    
    gr.Markdown("<hr>")  # Separator line
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="❓ Ask a Question", placeholder="Type your question here...")
            ask_button = gr.Button("Ask", elem_id="ask_button")
        answer_output = gr.Textbox(label="Response", interactive=False)

    # Button actions
    upload_button.click(process_and_load_document, inputs=[file_input], outputs=[upload_status])
    ask_button.click(ask_question, inputs=[query_input], outputs=[answer_output])

    # Add some CSS for styling
    gr.HTML(""" 
    <style> 
        body {
            background-color: #121212;  /* Dark background */
            font-family: Arial, sans-serif;
            color: #fff;  /* White text color for readability */
        }
        #upload_button { 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 10px 0; 
            cursor: pointer; 
            border-radius: 5px; 
        } 
        #upload_button:hover { 
            background-color: #45a049; 
        } 
        #ask_button { 
            background-color: #008CBA; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 10px 0; 
            cursor: pointer; 
            border-radius: 5px; 
        } 
        #ask_button:hover { 
            background-color: #007bb5; 
        } 
        .gradio-container {
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #1c1c1c;  /* Dark container */
        }
        .gradio-textbox input {
            color: #fff;  /* White text for textboxes */
            background-color: #333;  /* Dark background for textboxes */
        }
    </style> 
    """)

# Launch the Gradio app with localhost and port
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Render uses the PORT environment variable
    app.launch(server_name="0.0.0.0", server_port=port)

# import os
# import gradio as gr
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv
# from groq import Groq
# from langchain_groq import ChatGroq

# # Load environment variables
# load_dotenv()

# # Set up Groq client
# api_key = os.getenv("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("GROQ_API_KEY not found in .env file. Please set it up.")
# client = Groq(api_key=api_key)
# llm = ChatGroq(model_name='llama3-70b-8192')

# # Initialize global variables
# retriever = None
# qa_chain = None

# def process_and_load_document(file):
#     """Loads, splits, and embeds the uploaded document."""
#     global retriever, qa_chain

#     # Load document
#     loader = PyPDFLoader(file.name)
#     data = loader.load()

#     # Split document into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     text = text_splitter.split_documents(data)

#     # Create embeddings and vector store
#     embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
#     db = FAISS.from_documents(text, embeddings)

#     # Setup retriever
#     retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#     # Setup QA chain
#     prompt_template = '''
#     You are a helpful assistant. Greet the user before answering.

#     {context}
#     {question}
#     '''
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                            chain_type="stuff",
#                                            retriever=retriever,
#                                            chain_type_kwargs={"prompt": prompt},
#                                            return_source_documents=True)
#     return "📄 Document uploaded and processed successfully!"

# def ask_question(query):
#     """Handles user queries by leveraging the QA chain."""
#     if not qa_chain:
#         return "⚠️ Please upload a document first."
#     result = qa_chain(query)
#     return f"💬 **Answer:** {result['result']}"

# # Gradio Interface
# with gr.Blocks() as app:
#     gr.Markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat with Documents AI</h1>")
#     gr.Markdown(
#         "<p style='text-align: center; color: #555;'>Upload a document and ask questions to interact with its content!</p>"
#     )
    
#     with gr.Row():
#         with gr.Column():
#             file_input = gr.File(label="📁 Upload PDF Document", type="filepath")
#             upload_button = gr.Button("Upload and Process", elem_id="upload_button")
#         upload_status = gr.Textbox(label="Upload Status", interactive=False)
    
#     gr.Markdown("<hr>")  # Separator line
    
#     with gr.Row():
#         with gr.Column():
#             query_input = gr.Textbox(label="❓ Ask a Question", placeholder="Type your question here...")
#             ask_button = gr.Button("Ask", elem_id="ask_button")
#         answer_output = gr.Textbox(label="Response", interactive=False)

#     # Button actions
#     upload_button.click(process_and_load_document, inputs=[file_input], outputs=[upload_status])
#     ask_button.click(ask_question, inputs=[query_input], outputs=[answer_output])

#     # Add some CSS for styling
#     gr.HTML("""
#     <style>
#         #upload_button { 
#             background-color: #4CAF50; 
#             color: white; 
#             border: none; 
#             padding: 10px 20px; 
#             text-align: center; 
#             text-decoration: none; 
#             display: inline-block; 
#             font-size: 16px; 
#             margin: 10px 0; 
#             cursor: pointer; 
#             border-radius: 5px;
#         }
#         #upload_button:hover {
#             background-color: #45a049;
#         }
#         #ask_button {
#             background-color: #008CBA;
#             color: white;
#             border: none;
#             padding: 10px 20px;
#             text-align: center;
#             text-decoration: none;
#             display: inline-block;
#             font-size: 16px;
#             margin: 10px 0;
#             cursor: pointer;
#             border-radius: 5px;
#         }
#         #ask_button:hover {
#             background-color: #007bb5;
#         }
#     </style>
#     """)

# # Launch the Gradio app
# if __name__ == "__main__":
#     app.launch()


