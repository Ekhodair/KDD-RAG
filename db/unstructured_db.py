import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

import pandas as pd
from elasticsearch import AsyncElasticsearch
from langchain_elasticsearch import AsyncElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain_elasticsearch.vectorstores import ApproxRetrievalStrategy, DistanceStrategy
from dotenv import load_dotenv
sys.path.append("helpers")
from logger import create_logger


logger = create_logger(__name__)


class ElasticsearchRetrievalManager:
    
    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "job_product_index",
        embedding_model: str = "BAAI/bge-m3"
    ):
        self.es_url = es_url
        self.index_name = index_name
        self.es_client = AsyncElasticsearch(es_url)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda'}
        )
        self.vector_store = AsyncElasticsearchStore(
            es_connection=self.es_client,
            index_name=self.index_name,
            embedding=self.embeddings,
            vector_query_field='product_jobs_vector',
            query_field='product_jobs_text',
            distance_strategy=DistanceStrategy.COSINE,
            strategy=ApproxRetrievalStrategy(hybrid=True, rrf=False)
        )
    
    async def check_index_exists(self, index_name: str = None) -> bool:
        """Check if an index exists in Elasticsearch."""
        index_name = index_name or self.index_name
        return await self.es_client.indices.exists(index=index_name)
    
    async def delete_index(self, index_name: str = None) -> bool:
        """Delete an index from Elasticsearch."""
        index_name = index_name or self.index_name
        if await self.check_index_exists(index_name):
            await self.es_client.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")
            return True
        return False
    
    def process_csv_to_documents(self, csv_path: str) -> List[Document]:
        """
        Process a CSV file into a list of Document objects.
        """
        df = pd.read_csv(csv_path)
        source_file = os.path.basename(csv_path)
        documents = []
        for _, row in df.iterrows():
            row_data = row.fillna('').to_dict()
            row_data['source'] = source_file
            page_content = " ".join([f"{k}: {v}" for k, v in row_data.items() if v])
            metadata = {k: v for k, v in row_data.items() if k != 'description'}
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Processed {len(documents)} records from {source_file}")
        return documents
    
    async def index_documents(self, documents: List[Document]):
        """
        Index documents in Elasticsearch using LangChain's vector store.
        """
       
        await self.vector_store.aadd_documents(documents)
        logger.info(f"Successfully Indexed {len(documents)} documents")
    
    async def delete_document(self, document_id: str, index_name: str = None):
        """
        Delete a specific document from the index by ID.
        """
        index_name = index_name or self.index_name
        try:
            await self.es_client.delete(index=index_name, id=document_id)
            logger.info(f"Deleted document with ID: {document_id} from index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document with ID {document_id}: {str(e)}")
            return False
            
    async def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
        """
        Search the index using the hybrid similarity search.
        """
        try:
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=top_k,
                filter=filter_dict
            )
            results = '\n'.join([doc.page_content for doc in results])
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return ""

    async def close(self):
        """Close the Elasticsearch client connection."""
        await self.es_client.close()


class ElasticsearchHistoryManager:
    """Class for managing chat history in Elasticsearch."""
    
    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        chat_history_index: str = "chat_history"
    ):
        self.es_url = es_url
        self.chat_history_index = chat_history_index
        self.es_client = AsyncElasticsearch(es_url)
    
    async def check_index_exists(self, index_name) -> bool:
        """Check if an index exists in Elasticsearch."""
        index_name = self.chat_history_index
        return await self.es_client.indices.exists(index=index_name)
    
    async def create_chat_history_index(self):
        """Create the chat history index if it doesn't exist."""
        if not await self.check_index_exists(self.chat_history_index):
            await self.es_client.indices.create(
                index=self.chat_history_index,
                body={
                    "mappings": {
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "history": {"type": "text"},
                        }
                    }
                }
            )
            logger.info(f"Created chat history index: {self.chat_history_index}")
    
    async def store_chat_history(self, session_id: str, messages: List[Dict[str, str]]):
        """
        Store chat history in Elasticsearch.
        """
        # await self.create_chat_history_index()
        history_str = json.dumps(messages)
        doc = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "history": history_str
        }
        
        await self.es_client.update(
            index=self.chat_history_index,
            id=session_id,
            body={
                "doc": doc,
                "doc_as_upsert": True
            },
            refresh=True)
        logger.info(f"Stored chat history for session: {session_id}")
    
    async def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve chat history from Elasticsearch.
        """
        if not await self.check_index_exists(self.chat_history_index):
            logger.warning(f"Chat history index {self.chat_history_index} does not exist")
            return []

        try:
            response = await self.es_client.get(
                index=self.chat_history_index,
                id=session_id
            )
            history_str = response["_source"]["history"]
            return json.loads(history_str)

        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 404:
                logger.info(f"No chat history found for session: {session_id}")
            else:
                logger.error(f"Error retrieving chat history for session {session_id}: {str(e)}")
            return []

    
    async def delete_chat_history(self, session_id: str) -> bool:
        """
        Delete chat history for a specific session.
        """
        try:
            await self.es_client.delete_by_query(
                index=self.chat_history_index,
                body={
                    "query": {"term": {"session_id": session_id}}
                },
                refresh=True
            )
            logger.info(f"Deleted chat history for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chat history for session {session_id}: {str(e)}")
            return False
    
    async def close(self):
        """Close the Elasticsearch client connection."""
        await self.es_client.close()


async def main_async():
    """Async main function to run the indexer from command line."""
    parser = argparse.ArgumentParser(description="Index CSV data in Elasticsearch using LangChain")
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Directory containing CSV files to index"
    )
    
    args = parser.parse_args()
    
    # Initialize search manager
    search_manager = ElasticsearchRetrievalManager()
    
    # Initialize chat history manager
    chat_history_manager = ElasticsearchHistoryManager()
    
    csv_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv')]
    total_documents = 0
    
    for csv_filename in csv_files:
        csv_path = os.path.join(args.data_dir, csv_filename)
        
        logger.info(f"Processing CSV file: {csv_path}")
        documents = search_manager.process_csv_to_documents(csv_path)
        
        logger.info(f"Indexing {len(documents)} documents from {csv_filename}...")
        await search_manager.index_documents(documents)
        
        total_documents += len(documents)
    
    logger.info(f"Indexing completed successfully. Total documents indexed: {total_documents}")
    
    # Create chat history index 
    await chat_history_manager.create_chat_history_index()
    
    # Close connections
    await search_manager.close()
    await chat_history_manager.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()