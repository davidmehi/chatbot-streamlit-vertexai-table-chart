
# Chatbot Example with Streamlit, VertexAI, Chroma and UI Elements (Tables, Charts, Graphs)

This is a quick example of how to build a chatbot that can generate dynamic charts and graphs based on data from an LLM.  Note that this is not optimized or tuned.  There are some hardcoded logic for the purposes of the demo.

## Example prompts

```
Give me the major market sectors of the fidelity total market index fund as a table
Give me the major market sectors of the fidelity total market index fund as a bar chart
Give me the top stocks
Give me the performance of all years
```

## How it works

 * The UI is prototyped with Streamlit
 * The datasource `fidelity-total-market-index-fund.pdf` is indexed using Langchain.  It will read the PDF, generate embeddings, and stored in the in-memory Chroma vector database. 
 * Langchain facilitates A prompt that is sent to the LLM using the datasource `fidelity-total-market-index-fund.pdf` as the frame of reference.  
 * The app identifies that the user is asking for a table or chart.  If they are, then modify the prompt to ask for the data in csv format.  
 * Retrieve the data as csv from the LLM call
 * Read the csv data into a python Dataframe.  There might be some massaging of the data needed.  
 * Feed the Dataframe into a UI element 
   * Table is using [AgGrid](https://www.ag-grid.com/)
   * Chart is using matplotlib.  But there are many other charting libraries out there, including Google Charts.  
   * UI widgets have advanced functionality build into the widget.  Once the data is fed into the widget, the widget takes it from there and the user can manipulate and play with the data.
   * This part will take some custom programming.  
   * An alternative would be calling the VertexAI Codekey model to have the model generate python code to create a chart.  Run the python code and then generate the image.  Display the image in the UI.
 * Display the UI element in the UI 


## Install Libraries

```
pip3 install google-cloud-aiplatform --upgrade --user
pip3 install langchain --upgrade
pip3 install beautifulsoup4
pip3 install docarray
pip3 install tiktoken
pip3 install streamlit
pip3 install pandas datasets pypdf
# Install Vertex AI LLM SDK, langchain and dependencies
pip3 install google-cloud-aiplatform chromadb==0.3.26 pydantic==1.10.8 typing-inspect==0.8.0 typing_extensions==4.5.0 pandas datasets google-api-python-client pypdf faiss-cpu transformers config --upgrade --user
pip3 install streamlit-aggrid
pip3 install pandas
```

## Run with Streamlit

```
pip install streamlit
streamlit run chatbot.py
```
