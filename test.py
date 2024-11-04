import os
import getpass
from neo4j import GraphDatabase, Result
import pandas as pd
import tiktoken
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from llama_index.core.schema import TextNode
# from llama_index.core.vector_stores.utils import node_to_metadata_dict
# from llama_index.core.vector_stores.neo4jvector import Neo4jVectorStore
# from llama_index.core import VectorStoreIndex
from tqdm import tqdm


from typing import Dict, Any
from langchain_openai import OpenAI, ChatOpenAI

#sp key
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

llm_openai = ChatOpenAI(model="gpt-4")


# # Adjust pandas display settings
# pd.set_option(
#     "display.max_colwidth", None
# )  # Set to None to display the full column width

# pd.set_option("display.max_columns", None)

NEO4J_URI=os.getenv('NEO4J_URI')
NEO4J_USERNAME=os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
#     """Executes a Cypher statement and returns a DataFrame"""
#     return driver.execute_query(
#         cypher, parameters_=params, result_transformer_=Result.to_df
#     )

# res = db_query("MATCH (n) RETURN n LIMIT 5")
# print(res)

topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

from langchain_openai import OpenAIEmbeddings

embed = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# MAP_SYSTEM_PROMPT = """
# ---Role---

# You are a helpful assistant responding to questions about data in the tables provided.


# ---Goal---

# Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

# You should use the data provided in the data tables below as the primary context for generating the response.
# If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# Each key point in the response should have the following element:
# - Description: A comprehensive description of the point.
# - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

# The response should be JSON formatted as follows:
# {{
#     "points": [
#         {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
#         {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
#     ]
# }}

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

# Points supported by data should list the relevant reports as references as follows:
# "This is an example sentence supported by data references [Data: Reports (report ids)]"

# **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

# For example:
# "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

# where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

# Do not include information where the supporting evidence for it is not provided.


# ---Data tables---

# {context_data}

# ---Goal---

# Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

# You should use the data provided in the data tables below as the primary context for generating the response.
# If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# Each key point in the response should have the following element:
# - Description: A comprehensive description of the point.
# - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

# Points supported by data should list the relevant reports as references as follows:
# "This is an example sentence supported by data references [Data: Reports (report ids)]"

# **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

# For example:
# "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

# where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

# Do not include information where the supporting evidence for it is not provided.

# The response should be JSON formatted as follows:
# {{
#     "points": [
#         {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
#         {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
#     ]
# }}
# """

# map_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             MAP_SYSTEM_PROMPT,
#         ),
#         (
#             "human",
#             "{question}",
#         ),
#     ]
# )

# map_chain = map_prompt | llm_openai | StrOutputParser()

# REDUCE_SYSTEM_PROMPT = """
# ---Role---

# You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

# Note that the analysts' reports provided below are ranked in the **descending order of importance**.

# If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

# The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

# **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

# For example:

# "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

# where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

# Do not include information where the supporting evidence for it is not provided.


# ---Target response length and format---

# {response_type}


# ---Analyst Reports---

# {report_data}


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

# Note that the analysts' reports provided below are ranked in the **descending order of importance**.

# If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

# The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

# **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

# For example:

# "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

# where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

# Do not include information where the supporting evidence for it is not provided.


# ---Target response length and format---

# {response_type}

# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
# """

# reduce_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             REDUCE_SYSTEM_PROMPT,
#         ),
#         (
#             "human",
#             "{question}",
#         ),
#     ]
# )
# reduce_chain = reduce_prompt | llm_openai | StrOutputParser()

# graph = Neo4jGraph(
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD,
#     refresh_schema=False,
# )

# response_type: str = "multiple paragraphs"


# def global_retriever(query: str, level: int, response_type: str = response_type) -> str:
#     community_data = graph.query(
#         """
#     MATCH (c:__Community__)
#     WHERE c.level = $level
#     RETURN c.full_content AS output
#     """,
#         params={"level": level},
#     )
#     intermediate_results = []
#     for community in tqdm(community_data, desc="Processing communities"):
#         intermediate_response = map_chain.invoke(
#             {"question": query, "context_data": community["output"]}
#         )
#         intermediate_results.append(intermediate_response)
#     final_response = reduce_chain.invoke(
#         {
#             "report_data": intermediate_results,
#             "question": query,
#             "response_type": response_type,
#         }
#     )
#     return final_response



lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m)
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m)
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping,
       Relationships: outsideRels + insideRels,
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

index_name = "entity"

db_query(
    """
CREATE VECTOR INDEX """
    + index_name
    + """ IF NOT EXISTS FOR (e:__Entity__) ON e.description_embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 3072,
 `vector.similarity_function`: 'cosine'
}}
"""
)

db_query(
    """
MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
WITH n, count(distinct c) AS chunkCount
SET n.weight = chunkCount"""
)

lc_vector = Neo4jVector.from_existing_index(
    embed,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=index_name,
    retrieval_query=lc_retrieval_query,
)

docs = lc_vector.similarity_search(
    "Give me people with experience in python",
    k=topEntities,
    params={
        "topChunks": topChunks,
        "topCommunities": topCommunities,
        "topOutsideRels": topOutsideRels,
        "topInsideRels": topInsideRels,
    },
)
print(docs)


output_file = 'retriever_output.txt'

# Example: Assuming docs[0].page_content is the content you want to write
content = str(docs)

# Check if the file exists
if os.path.exists(output_file):
    mode = 'a'  # Append if the file exists
else:
    mode = 'w'  # Write if the file doesn't exist (create new file)

# Open the file in the appropriate mode and write the content
with open(output_file, mode) as file:
    file.write(content + '\n')  # Add a newline after writing

