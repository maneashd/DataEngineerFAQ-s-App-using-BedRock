# Data Engineering FAQ's with RAG using Bedrock
![Project Architecture](ScreenShots/Screenshot6.png)

## Overview
This Project depicts the easy and programmatical approach to create a User intterface where a user can ask questions regarding Data Engineering using Large Language Models(LLM's) using Langchain framework and Amazon BedRock!

## Technologies Used
- Langchain // Framework
- Amazon BedRock // Amazon Service to access all LLM's
- Streamlit // Develop FrontEnd UI for interaction.


# How to run the application.

- conda create -n llmapp python=3.8 -y //Create a new environment
- conda activate llmapp //Activate the environment
- pip install -r requirements.txt //Install the requirements package
- pip install -U langchain-community //Install Langchain community
- streamlit run main.py //run application using Streamlit

# Packages needed to be imported.

```python
import boto3    # It is AWS client used to interact with Amazon Web Services.
import streamlit as st    # Used to develop FrontEnd for User interaction.
from langchain.llms.bedrock import Bedrock    # Amazon Service to access all LLM's using API.
from langchain.embeddings import BedrockEmbeddings    # To choose an embedding model among available in BedRock.
from langchain.document_loaders import PyPDFDirectoryLoader    # Used to load the PDF files to feed our KnowledgeBase.
from langchain.text_splitter import RecursiveCharacterTextSplitter    # This helps in creating the Chunks for our embedding model as it can take only limited tokens at once.
from langchain.vectorstores import FAISS    # Acts as a VectorDB/KnowledgeBase to store the vectors/embeddings which can later be accessed by our LLM.
from langchain.prompts import PromptTemplate    # Promot given to our LLM Application, refers how our application should work.
from langchain.chains import RetrievalQA    # Since this application is a question and answer type, we use RetrievalQA.
```




