Chat with Documents
A powerful AI-driven tool that enables users to interact with documents of various formats by uploading them and asking questions. This project leverages advanced language models, vector search, and OCR to extract, process, and retrieve information seamlessly.

🚀 Features
Multi-format Document Support:
📄 PDF
📜 Word Documents (.docx)
📝 Text Files (.txt)
📊 CSV Files (.csv)
🖼️ Image Files (.png, .jpg, .jpeg, .bmp, .tiff)
Interactive Q&A: Ask questions about the uploaded document's content.
State-of-the-art NLP: Powered by Groq AI and LangChain for intelligent responses.
Embeddings and Retrieval: Uses FAISS for efficient similarity-based document retrieval.
OCR Integration: Extracts text from images using Tesseract OCR.


🛠️ Installation
Prerequisites
Python: Version 3.8 or above.
Linux: Install via sudo apt install tesseract-ocr.
Windows: Download and install Tesseract.
Mac: Install via brew install tesseract.
Steps
Clone the repository:

bash
Copy code
git clone https://https://github.com/Championshi/chat-with-documents.git
cd chat-with-documents
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add your Groq API Key:

Create a .env file in the project root:
plaintext
Copy code
GROQ_API_KEY=your_groq_api_key
Run the application:

bash
Copy code
python app.py
Open the generated link to access the Gradio interface.

🧩 Usage
Upload a Document:
Upload a supported file format (PDF, DOCX, TXT, CSV, or Image).
Ask Questions:
Enter your query in the provided text box.
View Results:
The app retrieves and answers questions using the document's content.
🌟 Examples
Supported Files
File Type	Description
PDF	Research papers, reports, or scanned documents.
DOCX	Word documents such as resumes or project documentation.
TXT	Plain text files containing simple content.
CSV	Tabular data that can be converted into text for interaction.
Image	Screenshots or scanned images of text. Extracted using Tesseract OCR.
Sample Use Case
Upload a PDF manual.
Ask: "How do I troubleshoot error code 404?"
The app retrieves relevant content and generates an answer.
🔧 Technologies Used
Frontend: Gradio for an interactive UI.
NLP: LangChain and Groq AI for question answering.
Embeddings: FastEmbed with FAISS for similarity search.
OCR: Tesseract for text extraction from images.
📝 License
This project is licensed under the MIT License.

🤝 Contributing
Fork the repository.
Create a feature branch:
bash
Copy code
git checkout -b feature-name
Commit changes and push:
bash
Copy code
git commit -m "Add feature"
git push origin feature-name
Create a pull request.
📧 Contact
For queries or support, please reach out to:

Name: Abhinav Kumar
Email: your-abhi629941@gmail.com
