# KDD-RAG
***
# Overview
A chat service that allows to converse with KDD website content about products and careers. The system implements 3 different RAG approaches:

Adaptive RAG: Adjusts retrieval method based on query complexity.

Fusion RAG: Combines both syntactic and semantic retrieval methods.

Graph RAG: Leverages graph relationships between entities in scraped data.

## Design

### RAG Architecture

The system uses a simple interface-based design:

```
BaseRAG (interface)
  ├── AdaptiveRAG
  ├── FusionRAG
  └── GraphRAG
```

- **BaseRAG**: Defines required methods (`retrieve()` and `generate()`) that all RAG Types must provide.

- **AdaptiveRAG**: Classifies queries and selects the appropriate retrieval method:
  - No retrieval for generic/irrelevant topics.
  - Uses Elasticsearch for standard queries
  - Combines Elasticsearch and graph database for complex queries.

- **GraphRAG**: Uses graph database relationships for knowledge retrieval.

- **FusionRAG**: Combines semantic and sytactic retrieval.

Each implementation handles retrieval and response generation differently while following the same basic interface.
***
## Install

### Python Version
```sh
python-3.10.12
```

***

### Virtual environment 
**Please make sure to initialize a virtual environment before installing any requirements:**

    $ python3 -m venv .venv
    $ source .venv/bin/activate
    
### Requirements

    $ pip install -r requirements.txt


## RUN

### 1. Set Environment Variables

Set all required environment variables in the shell as shown in the `.env-example` file:

```bash
$ source .env
```

### 2. Start Databases

To run the databases:

Elasticsearch: Vector/text database (http://localhost:9200)

Kibana: Elasticsearch dashboard and monitoring (http://localhost:5601)

Neo4j: bolt://localhost:7687


```bash
$ docker-compose up -d
```

You can verify services are running with:

    $ docker-compose ps

### 3. Scrape Data

To scrape data from sources, specify an output directory to save the scraped data:

```bash
$ python scrape.py --output-dir ./scraped_data
```

### 4. Run LLM

Run AWQ 4 bit quantized version of Llama3.3 70B model using vLLM for efficient inference:

```bash
$ vllm serve casperhansen/llama-3.3-70b-instruct-awq --trust-remote-code --tensor-parallel-size 2 --gpu_memory_utilization 0.9 --max-model-len 80000
```

### 5. Index Data in Unstructured Database

Index the scraped data in ES database:

```bash
$ python db/unstructured_db.py --data-dir <dir containing scraped data>
```

### 6. Index Data in Graph Database

**To use pre-extracted graph data (faster):**

    $ python db/graph_db.py --data-dir ./graph_data --graph-ready

**To extract nodes/relationships and index graph data from scratch:**

    $ python db/graph_db.py --data-dir <scraped-files-dir>


### 7. Run the API Server

Start the RESTful API with the `/chat` endpoint:

```bash
$ uvicorn api:app --host 0.0.0.0 --port 8001
```

### 8. Evaluate the System

Run evaluation across all RAG types using the provided prompts:

```bash
$ python evaluate.py
```

Evaluation metrics include:
- **Relevance (0-10):** How directly the response addresses the query
- **System Persona Adherence (SPAR, 0-10):** Consistency in maintaining the defined professional persona
- **Chat History (CH, 0-5):** Accuracy in retaining and referencing conversation context
- **Response Quality (RQ, 0-10):** Overall coherence, fluency, and logical consistency
- **Latency:** Time from request to complete response

 ### Inference
  
To start an interactive chat with the /chat endpoint, please run
```bash
$ python interactive_chat.py
```

   
## Input / Output

### Chat Endpoint

**POST** `/chat`

#### Request Payload
```json
{
    "question": string,
    "rag_type": string,
    "session_id": string | null
}
```

#### Response
Server-Sent Events (SSE) stream with the following format for each chunk:
```json
{
    "token": string,
    "session_id": string
}
```



**You can reach and interact with the system through http://0.0.0.0:8001**

**The documentation for api can be accessed through http://0.0.0.0:8001/docs**
