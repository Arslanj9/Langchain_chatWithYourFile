📚 Chat with Your File (LangChain + HuggingFace)
This project lets you chat with any text file (e.g., articles, notes, documents) using LangChain, FAISS vector database, and a HuggingFace LLM.
It loads your file, splits it into chunks, creates embeddings, and answers your questions interactively.

🚀 Features
Load and process your own .txt file
Store embeddings locally with FAISS
Use HuggingFace LLMs for answering questions
Interactive chatbot in your terminal
Simple setup with .env file

🛠️ Installation

Clone this repo

git clone https://github.com/your-username/chat-with-file.git
cd chat-with-file


Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt

⚙️ Configuration

Create a .env file in the root folder:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here


Add a text file named sample.txt in the project folder.
This is the file you will chat with.


▶️ Usage

Run the chatbot:

python chat_with_file.py


Example:

🔍 Starting script...
✅ .env loaded
✅ HuggingFace API key found
✅ sample.txt found
✅ Chatbot is ready! Ask questions about your text (type 'exit' to quit).

You: Who is Einstein?
Bot: Albert Einstein was a theoretical physicist, best known for the theory of relativity.

📂 Project Structure
chat-with-file/
│── chat_with_file.py    # Main script
│── sample.txt           # Your text file (user-provided)
│── .env                 # HuggingFace token
│── requirements.txt     # Dependencies
│── README.md            # Project docs

📦 Dependencies

See requirements.txt:

python-dotenv
langchain
langchain-community
langchain-huggingface
faiss-cpu
transformers
huggingface-hub

🔮 Future Improvements

Support for PDF, Word, and CSV files
Web UI with Streamlit or Next.js
Multiple file support

📝 License

This project is open-source. Feel free to fork and improve 🚀
