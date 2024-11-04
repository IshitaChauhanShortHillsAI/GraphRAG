from graphdatascience import GraphDataScience
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

# project graph
url="bolt://localhost:7687",
username="neo4j", 
password="12345678"  

gds = GraphDataScience(
    os.getenv("NEO4J_DB_URL"),
    auth=(os.getenv("NEO4J_DB_USERNAME"), os.getenv("NEO4J_DB_PASSWORD"))
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_DB_URL"),
    username=os.getenv("NEO4J_DB_USERNAME"), 
    password=os.getenv("NEO4J_DB_PASSWORD")
)

# G, result = gds.graph.project(
#     "communities",  #  Graph name
#     "__Entity__",  #  Node projection
#     {
#         "_ALL_": {
#             "type": "*",
#             "orientation": "UNDIRECTED",
#             "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
#         }
#     },
# )

# res = gds.leiden.write(
#     G,
#     writeProperty="communities",
#     includeIntermediateCommunities=True,
#     relationshipWeightProperty="weight",
# )



# res = graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")
# print(res)

# res2 = graph.query("""
# MATCH (e:`__Entity__`)
# UNWIND range(0, size(e.communities) - 1 , 1) AS index
# CALL {
#   WITH e, index
#   WITH e, index
#   WHERE index = 0
#   MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
#   ON CREATE SET c.level = index
#   MERGE (e)-[:IN_COMMUNITY]->(c)
#   RETURN count(*) AS count_0
# }
# CALL {
#   WITH e, index
#   WITH e, index
#   WHERE index > 0
#   MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
#   ON CREATE SET current.level = index
#   MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
#   ON CREATE SET previous.level = index - 1
#   MERGE (previous)-[:IN_COMMUNITY]->(current)
#   RETURN count(*) AS count_1
# }
# RETURN count(*)
# """)

# print(res2)

# res3 = community_info = graph.query("""
# MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
# WHERE c.level IN [0,1,4]
# WITH c, collect(e ) AS nodes
# WHERE size(nodes) > 1
# CALL apoc.path.subgraphAll(nodes[0], {
# 	whitelistNodes:nodes
# })
# YIELD relationships
# RETURN c.id AS communityId,
#        [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
#        [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
# """)

# print(res3)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_openai import OpenAI, ChatOpenAI

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

llm_openai = ChatOpenAI(model="gpt-4")

community_template = """Based on the provided nodes and relationships that belong to the same graph community,
generate a natural language summary of the provided information:
{community_info}

Summary:"""  # noqa: E501

community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input triples, generate the information summary. No pre-amble.",
        ),
        ("human", community_template),
    ]
)

community_chain = community_prompt | llm_openai | StrOutputParser()
community_info = graph.query("""
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
WHERE c.level IN [0,1,4]
WITH c, collect(e ) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId,
       [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
""")

def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

def process_community(community):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary}

# print(prepare_string(community_info[3]))

import json


output_file = 'summaries.txt'  # Change this to your desired file path

summaries = []
with open(output_file, 'a') as file:  # Open the file in append mode
    for community in tqdm(community_info, total=len(community_info), desc="Processing communities"):
        summary = process_community(community)
        summaries.append(summary)
        
        # Convert the dictionary to a JSON string and write it to the file
        file.write(json.dumps(summary) + '\n')
