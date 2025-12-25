import chromadb
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_core.prompts import ChatPromptTemplate

# initialize GenAI client
gemini_api_key ="AIzaSyCU3bcujpYmPpvrkatvXbRwnlxI7pWgeow"
client = genai.Client(api_key=gemini_api_key)

# for this example I used Alphabet Inc 10-K Report 2022
# https://s206.q4cdn.com/479360582/files/doc_financials/2024/q4/goog-10-k-2024.pdf
DOC_PATH = "Summarise_pdf_data/goog-10-k-2024.pdf"
CHROMA_PATH = "Chroma"
# ----- Data Indexing Process -----
# load your pdf doc
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

# split the doc into smaller chunks i.e. chunk_size=500
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# get Gemini Embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",   # Gemini embedding model
    google_api_key=gemini_api_key
)


# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
# ----- Retrieval and Generation Process -----

# this is an example of a user question (query)
query = 'what are the top risks mentioned in the document?'

# retrieve context - top 5 most relevant (closests) chunks to the query vector
# (by default Langchain is using cosine distance metric)
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

# generate an answer based on given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
# you can use a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# load retrieved context and user query in the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# call LLM model to generate the answer based on the given context and query

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",          # or "gemini-1.5-flash" for faster responses
    google_api_key=gemini_api_key
)


response_text = model.predict(prompt)
print("Question:")
print(query)  
print("\nAnswer:")
print(response_text)  