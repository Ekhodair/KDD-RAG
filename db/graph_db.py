import sys
import re
import os
import json
import argparse
import pickle
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

import pandas as pd
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain.schema.document import Document
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

sys.path.append("helpers")
from prompts import GRAPH_SYSTEM_PROMPT, GRAPH_PROMPT, ENTITY_SYSTEM_PROMPT, ENTITY_PROMPT
from utils import call_model, clean_text
from logger import create_logger

logger = create_logger(__name__)

examples = [
            {
                "text": (
                    "Adam is a software engineer in Microsoft since 2009, "
                    "and last year he got an award as the Best Talent"
                ),
                "head": "Adam",
                "head_type": "Person",
                "relation": "WORKS_FOR",
                "tail": "Microsoft",
                "tail_type": "Company",
            },
            {
                "text": (
                    "Adam is a software engineer in Microsoft since 2009, "
                    "and last year he got an award as the Best Talent"
                ),
                "head": "Adam",
                "head_type": "Person",
                "relation": "HAS_AWARD",
                "tail": "Best Talent",
                "tail_type": "Award",
            }
        ]

class GraphDatabaseManager:
    
    def __init__(self, url='bolt://localhost:7687', username='neo4j', password='password'):
        self.graph = Neo4jGraph(url=url, username=username, password=password)
    
    @staticmethod
    def parse_model_response(response: str) -> List[Dict[str, str]]:
        """
        Parse the LLM's JSON response into structured relationships.
        """
        try:
            # Extract JSON array from the response using regex
            json_pattern = r'\[\s*{.*}\s*\]'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)                
                if isinstance(parsed, dict):
                    parsed = [parsed]
                
                return parsed
            else:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                return parsed
                
        except json.JSONDecodeError:
            print(f"Failed to parse response as JSON: {response}")
            return []

    @staticmethod
    def generate_full_text_query(input_text: str) -> str:
        """
        Generate a full-text search query for a given input string.
        """
        if not input_text:
            return ""
            
        words = [el for el in remove_lucene_chars(input_text).split() if el]
        
        if not words:
            return ""
            
        if len(words) == 1:
            return f"{words[0]}~2"
            
        full_text_query = ""
        for word in words[:-1]:
            full_text_query += f"{word}~2 AND "
        full_text_query += f"{words[-1]}~2"
        
        return full_text_query.strip()

    @staticmethod
    def init_graph_document(parsed_json: List[Dict[str, str]], document: str) -> GraphDocument:
        """
        Convert parsed JSON output to a GraphDocument object.
        """
        nodes_set: Set[Tuple[str, str]] = set()
        relationships = []
        
        for rel in parsed_json:
            # deduplicate using a set
            nodes_set.add((rel["head"], rel["head_type"]))
            nodes_set.add((rel["tail"], rel["tail_type"]))

            source_node = Node(id=rel["head"], type=rel["head_type"])
            target_node = Node(id=rel["tail"], type=rel["tail_type"])
            relationships.append(
                Relationship(
                    source=source_node, 
                    target=target_node, 
                    type=rel["relation"]
                )
            )
        # Create nodes list
        nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
        
        return GraphDocument(nodes=nodes, relationships=relationships, source=Document(page_content=document))
    
    def save_graph_data(self, graph_data: List[Dict], output_path: str) -> None:
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        print(f"Graph data saved to {output_path}")
    
    def load_graph_data(self, input_path: str) -> List[Dict]:
       
        with open(input_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        return graph_data
    
    def index(self, path: str, graph_ready: bool = True) -> None:
        
        
        base_filename = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join("graph_data", f"{base_filename}.json")
        
        graph_documents = []
        
        if graph_ready:
            graph_data = self.load_graph_data(output_path)
            
            # Convert loaded data to GraphDocument objects
            for item in tqdm(graph_data, desc="Converting saved data to GraphDocuments"):
                try:
                    text = item["text"]
                    parsed = item["parsed_relationships"]
                    graph_document = self.init_graph_document(parsed, text)
                    graph_documents.append(graph_document)
                except Execption as e:
                    print("Failed to process this document due to", e)
                    continue
        else:
            df = pd.read_csv(path)
            graph_data = []
            
            for _, row in tqdm(df.head(3).iterrows(), total=len(df.head(3)), desc="Processing rows"):
                text = " ".join([str(value) for value in row.values if pd.notna(value)])            
                messages = [
                    {'role': 'system', 'content': GRAPH_SYSTEM_PROMPT},
                    {'role': 'user', 'content': GRAPH_PROMPT.format(examples=json.dumps(examples), input=text)}
                ]
                try:
                    output = call_model(messages, max_tokens=1000)
                    # Parse model response
                    parsed = self.parse_model_response(output)                     
                    graph_data.append({
                        "text": text,
                        "parsed_relationships": parsed
                    })
                    
                    # Convert to GraphDocument
                    graph_document = self.init_graph_document(parsed, text)
                    graph_documents.append(graph_document)
                except Exception as e:
                    print(f"Failed due to {e}")
                    continue
        
            # Save the extracted data
            self.save_graph_data(graph_data, output_path)
            
        # Store to Neo4j
        self.graph.add_graph_documents(
            graph_documents, 
            baseEntityLabel=True, 
            include_source=True
        )
        
        print(f"Successfully indexed {len(graph_documents)} documents into Neo4j.")
    
    @staticmethod
    def extract_entities(query: str) -> List[str]:
        """
        Extract entities from input text using LLM.
        """
        messages = [
            {'role': 'system', 'content': ENTITY_SYSTEM_PROMPT},
            {'role': 'user', 'content': ENTITY_PROMPT.format(query=query)}
        ]
        
        response = call_model(messages, max_tokens=50, stream=False)
        return [clean_text(name.strip()) for name in response.split(',') if name.strip()]


    async def search(self, query: str) -> str:
        """
        Search the graph database using entities extracted from the query.
        """
        entities = self.extract_entities(query)
        result = ""
        
        for entity in entities:
            response = self.graph.query(
                """
                MATCH (node:__Entity__)
                WHERE node.id CONTAINS $query
                WITH node LIMIT 4
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    WHERE type(r) <> 'MENTIONS'
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    WHERE type(r) <> 'MENTIONS'
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])
            result += "\n"
        return result

def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description="Index CSV data in Neo4j graph database")
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Directory containing CSV or JSON graph files to index"
    )
    
    parser.add_argument(
        "--graph-ready",
        action="store_true",
        help="Skip LLM extraction and use saved graph data"
    )
    
    args = parser.parse_args()
    
    manager = GraphDatabaseManager()

    paths = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv') or f.lower().endswith('.json')]    
    for fname in paths:
        fpath = os.path.join(args.data_dir, fname)
        
        print(f"Processing CSV file: {fpath}")
        manager.index(fpath, graph_ready=args.graph_ready)    

if __name__ == "__main__":
    main()