

# import libraries
import streamlit as st
from google.cloud import aiplatform

# Utils
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatVertexAI
import time
from typing import List

# LangChain
import langchain
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Display
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io
from io import StringIO
import streamlit_gchart as gchart
from PIL import Image
import tempfile

print(f"Vertex AI SDK version: {aiplatform.__version__}")
print(f"LangChain version: {langchain.__version__}")


# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]

# LLM model
llm = VertexAI(
    model_name="text-bison@latest",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# Chat
chat = ChatVertexAI()

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

# PDF Datasource
url = "fidelity-total-market-index-fund.pdf"
loader = PyPDFLoader(url)
documents = loader.load()

# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f"# of documents = {len(docs)}")

# load embeddings into the vector db
db = Chroma.from_documents(docs, embeddings)

# Build retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# Uses LLM to synthesize results from the search index.
# We use Vertex PaLM Text API for LLM
# Queries the LLM with the PDF document in mind
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

# Displays Table widget
def outputTable(data):
    # Column headers are hardcoded - should be getting it dynamic - todo
    data = 'Sector,Allocation\n' + data
    #df = pd.read_csv(data)
    df = pd.read_csv(io.StringIO(data), sep=",")
    AgGrid(df)

def outputChart(data):
    # Column headers are hardcoded and % removed - should be getting it dynamic - todo
    data = 'Sector,Allocation\n' + data
    data = data.replace("%","")
    print("\n\nCSV Data:")
    print(data)

    ## Massages the data for the Google Chart
    #data_lines = data.splitlines()
    #chart_data = []
    #for line in data_lines:
    #    line_data = line.split(",")
        #print(line_data)
    #    new_line_data = []
        # Convert numbers if they can, convert % to float
    #    for item in line_data:
    #        if "%" in item:
    #            item = item.replace("%","")
    #            print(item)
    #        try:
    #            item = float(item)
    #            new_line_data.append(item)
    #        except ValueError:
    #            new_line_data.append(item)

    #   chart_data.append(new_line_data)
    #   print(new_line_data)

    #print((chart_data))
    #st.subheader("Bar Chart")    

    #gchart.gchart(key="", data=chart_data, chartType="BarChart", width='500px', height='600px', 
    #    title="", hAxis={"title": "", "minValue": 0,"textPosition": "in"}, vAxis={"title": ""}, chartArea={"top": 55, "width": '40%'} )
    
    # Reads CSV data from the LLM into a Dataframe
    df = pd.read_csv(io.StringIO(data), sep=",")
    print(df)

    # Formats the chart
    rcParams.update({'figure.autolayout': True})
    plt.autoscale()

    # Need to dynamically pull label names - todo
    df.plot.bar(x='Sector', y='Allocation')
    temp_image_file = tempfile.TemporaryFile()
    plt.savefig(temp_image_file, bbox_inches="tight")
    image = Image.open(temp_image_file)

    # Display image
    st.image(image)


def query_llm(query):
    print(f"--querying---{query}")
    response = qa({"query": query})
    return response

st.title('Chatbot Q&A')
  
# Initialize with sample question
question=st.text_input("Ask a question","Give me the major market sectors of the fidelity total market index fund as a bar chart")

if st.button("Ask"):
        if question:
            
            if('table' in question):
                question = question.replace("table","csv format with column headers")
                print(question)
                response = query_llm(question)
                print(response)
                result = response["result"]
                outputTable(result)
            elif('chart' in question):
                question = question.replace("chart","csv format with column headers")
                print(question)
                response = query_llm(question)
                print(response)
                result = response["result"]
                outputChart(result)
            else:
                response = query_llm(question)
                print(response)
                result = response["result"]
                st.write(result)
        else:
            st.warning("Please enter a question.")
else:
    st.write(chat([HumanMessage(content="Hello")]).content)
