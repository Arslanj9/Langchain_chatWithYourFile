# chat_with_file.py

import os
import traceback
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document


print("🔍 Starting script...")

# 1. Load environment variables
load_dotenv()
print("✅ .env loaded")

# 2. Check Hugging Face API token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise ValueError("❌ HuggingFace API key not found. Please add it in .env file.")
print("✅ HuggingFace API key found")

# 3. Check if sample.txt exists
if not os.path.exists("sample.txt"):
    raise FileNotFoundError("❌ sample.txt not found in the current folder!")
print("✅ sample.txt found")

# 4. Load your text file
with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

if not text.strip():
    raise ValueError("❌ sample.txt is empty. Please add some text to it.")

# 5. Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

if not chunks:
    raise ValueError("❌ No chunks were created from the text. Check the file content.")

# 6. Convert chunks into LangChain Documents
documents = [Document(page_content=chunk) for chunk in chunks]

# 7. Create embeddings with Hugging Face
print("🔄 Creating embeddings...")
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)

# 8. Create Hugging Face LLM
print("🔄 Loading LLM...")
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",  # free instruct model
    temperature=0.5,
    max_new_tokens=200,
)

# 9. Setup prompt
system_prompt = (
    "You are a helpful assistant. Use the following context to answer the question. "
    "If you don’t know the answer, just say you don’t know. "
    "Keep answers concise (max 3 sentences).\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 10. Build retrieval chain
retriever = db.as_retriever()
doc_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)

print("✅ Chatbot is ready! Ask questions about your text (type 'exit' to quit).\n")

# 11. Interactive Q&A loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting. Goodbye!")
        break

    try:
        # Debug: see retrieved docs (new API)
        docs = retriever.invoke(query)
        print("\n🔎 Retrieved Contexts:")
        if not docs:
            print("  ⚠️ No relevant documents found.\n")
        else:
            for i, d in enumerate(docs, 1):
                print(f"  {i}. {d.page_content[:200]}...\n")


        # Run QA chain
        output = qa_chain.invoke({"input": query})
        print("🤖 Bot:", output["answer"], "\n")

    except Exception as e:
        print("❌ Error while processing query:")
        traceback.print_exc()  # full error trace
        print()
