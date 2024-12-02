import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
YOUR_ACCESS_KEY = os.getenv("YOUR_ACCESS_KEY")
YOUR_SECRET_KEY = os.getenv("YOUR_SECRET_KEY")
REGION_NAME = os.getenv("REGION_NAME")


# 1.1 Create a PromptTemplate for the GenAI app.

prompt_template = """

Human: Use the folowing pieces of the context to provide a
cocise answer to the question at the end also use atleast 250 words to summarize
and explain your answer in detail. If you do not know the answer, just say that 
"Can't answer this question with the knowledge I have", but do not try to make up answers.
<context>
{context}
</context>

Question: {question}

Assistant:"""

# 1.2 Initialize the prompt template.
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



# 2.Bedrock client setup.
bedrock = boto3.client(
    service_name='bedrock-runtime', 
    region_name=REGION_NAME,
    aws_access_key_id=YOUR_ACCESS_KEY,
    aws_secret_access_key=YOUR_SECRET_KEY)


# 3.Get embeddings model from Bedrock.
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# 4.Create Chunks to feed into embedding model.
def get_documents():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000, 
                                    chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs

# 5.get the vectors/embeddings from the chunks and save the vectors/embedding/vectorDB to local to use it for later.
def get_embeddings(docs):
    vectoestore_faiss = FAISS.from_documents(
                                    docs,
                                    bedrock_embedding
    ) 
    vectoestore_faiss.save_local("faiss_local")

# 6. Define an llm
def get_llm():
    llm = Bedrock(model_id = "mistral.mistral-7b-instruct-v0:2", client = bedrock)
    return llm

# 7. Get the llm response

def get_llm_response(llm, vectoestore_faiss, query):
    #The RetrievalQA will help to connect the llm to the knowledge base and also the query user asked.
    qa = RetrievalQA.from_chain_type(
         llm = llm,
         chain_type = "stuff", # there are multiple chain types like "stuff", "refine" etc.
         retriever = vectoestore_faiss.as_retriever(
         search_type="similarity", search_kwargs={"k": 3}), # if we user vecotDB we need to use similarity search.

         return_source_documents=True, # this will return the documents which contain the answer
         chain_type_kwargs={"prompt": PROMPT})

    response = qa({"query": query})
    return response['result']

def main():
    st.set_page_config("RAG")
    st.header("Data Engineering FAQ's with RAG using Bedrock")

    user_question = st.text_input("Ask a Question regarding Data Engineering")

    with st.sidebar:
        st.title("Create Knowledge Base once! Ask questions anytime!")

        if st.button("Store Vector"):
            with st.spinner("processing..."):
                docs = get_documents()
                get_embeddings(docs)
                st.success("Done")

        if st.button("Send"):
            with st.spinner("processing..."):
                faiss_index = FAISS.load_local("faiss_local",
                                               bedrock_embedding, 
                                               allow_dangerous_deserialization=True)
                llm = get_llm()
                st.write(get_llm_response(llm, faiss_index, user_question))




if __name__ == "__main__":
    main()
