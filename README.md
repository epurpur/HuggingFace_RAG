# Langchain RAG Tutorial

## Activate Virtual Environment
```
conda activate rag-env
```

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

## Create database

Create the Chroma DB.

```python
python create_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

> You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).

## Difference between Generative and Embedding model (for my understanding)
This program is using an embedding model, not a generative model. The difference is the embedding model does not generate new text while a generative AI model (such as GPT) does create new text. The embedding model searches my data for relevant snippets, converts them to a numerical vector (which captures the meaning), and then gives the response as the most relevant snippets loosely packaged as a response.

all-MiniLM-L6-v2 is the embedding part of the model that finds the most relevant chunks of text to feed the writer
FLAN-T5 is the generative part of the model that writes the answer


Questions asked:
- What are the services provided by Research Data Services?
    -response: decent
- What are some of the workshops provided by Research Data Services?
    -response: decent
- What should I know about funding agency requirements?
    -response: bad
- What are the options for data storage?
    -response: bad
- What are some of the licensed data sources offered?
    -response: bad
- What is the media intelligence center?
    -response: good

