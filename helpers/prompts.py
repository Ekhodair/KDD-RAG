# graph construction prompts
GRAPH_SYSTEM_PROMPT = """
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
Try to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.
- **Nodes** represent entities and concepts.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it
accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use available types for node labels.
Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms like 'mathematician' or 'scientist'.
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
- **Relationships** represent connections between entities or concepts. 
Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!
## 3. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
## 5. Output Format
You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type".
The "head" key must contain the text of the extracted entity.
The "head_type" key must contain the type of the extracted head entity.
The "relation" key must contain the type of relation between the "head" and the "tail".
The "tail" key must represent the text of an extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail entity.
Attempt to extract as many entities and relations as you can.
"""
GRAPH_PROMPT = """Below are a number of examples of text and their extracted 
        entities and relationships.
        {examples}\n"
        For the following text, extract entities and relations as 
        in the provided example.
        Text: {input}"""
# entity extraction prompts
ENTITY_SYSTEM_PROMPT = """You are expert at extracting entities from the text. these entities will be used for graph data retrieval.
You must generate the output as a python list of strings, where each element represents an extracted entity. Do not output any reasoning, Provide the list only."""

ENTITY_PROMPT = """"Use the given format to extract entities from the following input: {query}"""
# generation prompts
GENERATION_SYSTEM_PROMPT = "You are a helpful AI assistant. Use the given context to answer the user's question."
GENERATION_PROMPT = """Answer the question below from the following context:
### Context: {context}
### question: {question}
"""
# classifier prompts
CLASSIFIER_SYSTEM_PROMPT = """You are an expert in classifying queries by their relevance to a dataset containing information about jobs and products.
        Classify the input query into exactly one of the following categories:
        - **Irrelevant**: The query is unrelated to jobs or products. Includes greetings, casual conversation, or meta-questions about the system.
        - **Relevant**: The query directly concerns jobs or products and can be answered using the indexed data.
        - **Complex**: The query is related to jobs or products but requires multi-step reasoning, comparisons, or advanced filtering.

        Respond with only the category name: `Irrelevant`, `Relevant`, or `Complex`. Do not include any explanation or additional text.
        """
CLASSIFIER_PROMPT = """Classify this query: {query}"""

## Eval prompts
EVAL_AGENT_SYS_PROMPT = """You are an evaluation assistant for conversational AI system. Ensure your evaluation align with the provided metric definitions."""
EVAL_AGENT_PROMPT = """Your task is to critically evaluate the assistant's responses based on specific metrics.

Provide the final evaluation results clearly and concisely in the exact format below:

[[{{
  "Relevance": ,
  "SPAR": ,
  "CH": ,
  "RQ": ,
}}]]

**Metric Descriptions and Evaluation Process:**

1. **Relevance (0-10):**  
   Measures how directly the assistant's response addresses the user's query. It ranges from 0 to 10. Provide a numerical score along with justification.

2. **System Persona Adherence Ratio (SPAR, 0-10):**  
  Measures how consistently the assistant maintained its defined persona as a knowledgeable helper regarding jobs and products, ensuring relevant answer, professional tone, and topic focus. Justify your score.
3. **Chat History (CH, 0-5):**  
   Indicates whether the assistant correctly retained and referenced the chat history. It ranges from 0 to 5.

4. **Response Quality (RQ, 0-10):**  
   Assesses coherence, fluency, logical consistency, and context-awareness of the assistant across the entire interaction. It ranges from 0 to 10
   - Justify your score clearly.

### CHAT HISTORY START ###
{chat_history}
### CHAT HISTORY END ###

Now, carefully review the provided chat history, calculate each metric according to the guidelines, and deliver your assessment with thorough justifications in the exact dictionary format above.
"""
