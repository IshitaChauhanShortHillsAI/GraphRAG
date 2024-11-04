import os
from langchain_community.graphs import Neo4jGraph
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
import tiktoken
import pickle

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate

import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_community.chat_models import ChatLlamaCpp

from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USER"), 
    password=os.getenv("NEO4J_PASSWORD") 
)

os.environ['MISTRAL_API_KEY']=os.getenv('MISTRAL_API_KEY')

llm_mistral = ChatMistralAI(model="mistral-large-latest")

examples = [
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "STUDIED_AT",
        "tail": "Delhi Technological University",
        "tail_type": "College"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "PURSUING_COURSE",
        "tail": "Bachelor of Technology in software",
        "tail_type": "Course"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "82% in Class-XII",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "84% in Class-X",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "WORKED_AS",
        "tail": "Software Developer Intern",
        "tail_type": "Role"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "Aqeeq Technologies",
        "head_type": "Company",
        "relation": "EMPLOYED_BY",
        "tail": "akash kumar prasad",
        "tail_type": "Person"
    }
]

template = ChatPromptTemplate([
    ("system", """
Given a resume document and a list of entity types, identify the name of the person, their skills, and the courses they have completed. Additionally, identify any clear relationships between these entities.

Person names (e.g., Akansha, Aditi)
Programming language skills (e.g., Python, Java)
Course names (e.g., BTech, Phd)
Institute or college names (e.g., Indian Institute of Information Technology, Delhi)
Provide the entities in a clear and organized format.
     
-Steps-
1. Identify all relevant entities from the resume. For each identified entity, extract the following information:
- entity_name: Name of the person, skills, or course, capitalized
- entity_type: One of the following types: [Person, Skill, Course]
- entity_description: Comprehensive description of the entity's attributes and activities (for example, for a skill, mention the context or level of proficiency; for a course, mention the institution or time of completion)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why the source entity and the target entity are related (e.g., a skill that was acquired through a course)
- relationship_strength: an integer score between 1 to 10, indicating the strength of the relationship between the source entity and target entity

3. Return output in graph document as a single list of all the entities and relationships identified in steps 1 and 2. 
4. If you have to translate into graph document, just translate the descriptions, nothing else!


examples = [
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "STUDIED_AT",
        "tail": "Delhi Technological University",
        "tail_type": "College"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "PURSUING_COURSE",
        "tail": "Bachelor of Technology in software",
        "tail_type": "Course"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "82% in Class-XII",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "84% in Class-X",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "WORKED_AS",
        "tail": "Software Developer Intern",
        "tail_type": "Role"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "Aqeeq Technologies",
        "head_type": "Company",
        "relation": "EMPLOYED_BY",
        "tail": "akash kumar prasad",
        "tail_type": "Person"
    }
]"""),
    ("human", """entity_types: [Person, Skill, Course]"""),
])

system_prompt = """Analyze the following documents and list the different entities you find. Specifically, look for:
Person names (e.g., Akansha, Aditi)
Programming language skills (e.g., Python, Java)
Course names (e.g., BTech, Phd)
Institute or college names (e.g., Indian Institute of Information Technology, Delhi)
Provide the entities in a clear and organized format.

-Steps-
1. Identify all relevant entities from the resume. For each identified entity, extract the following information:
- entity_name: Name of the person, skills, or course, capitalized
- entity_type: One of the following types: [Person, Skill, Course, Institute/College name]
- entity_description: Comprehensive description of the entity's attributes and activities (for example, for a skill, mention the context or level of proficiency; for a course, mention the institution or time of completion)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why the source entity and the target entity are related (e.g., a skill that was acquired through a course)
- relationship_strength: an integer score between 1 to 10, indicating the strength of the relationship between the source entity and target entity

3. If you have to translate into graph document, just translate the descriptions, nothing else!

4. Donot create names or other information, strictly abide by the documentation given

Below are some examples:
[
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "STUDIED_AT",
        "tail": "Delhi Technological University",
        "tail_type": "College"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "PURSUING_COURSE",
        "tail": "Bachelor of Technology in software",
        "tail_type": "Course"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "82% in Class-XII",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "84% in Class-X",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "WORKED_AS",
        "tail": "Software Developer Intern",
        "tail_type": "Role"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "Aqeeq Technologies",
        "head_type": "Company",
        "relation": "EMPLOYED_BY",
        "tail": "akash kumar prasad",
        "tail_type": "Person"
    }
]
"""
template_2 = ChatPromptTemplate([('system',"""Analyze the following documents and list the different entities you find. Specifically, look for:
Person names (e.g., Akansha, Aditi)
Programming language skills (e.g., Python, Java)
Course names (e.g., BTech, Phd)
Institute or college names (e.g., Indian Institute of Information Technology, Delhi)
Provide the entities in a clear and organized format.

-Steps-
1. Identify all relevant entities from the resume. For each identified entity, extract the following information:
- entity_name: Name of the person, skills, or course, capitalized
- entity_type: One of the following types: [Person, Skill, Course]
- entity_description: Comprehensive description of the entity's attributes and activities (for example, for a skill, mention the context or level of proficiency; for a course, mention the institution or time of completion)
Format each entity as ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why the source entity and the target entity are related (e.g., a skill that was acquired through a course)
- relationship_strength: an integer score between 1 to 10, indicating the strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_strength>)

3. Return output in graph document as a single list of all the entities and relationships identified in steps 1 and 2. Use **{{record_delimiter}}** as the list delimiter.

4. If you have to translate into graph document, just translate the descriptions, nothing else!

5. When finished, output {{completion_delimiter}}.

Below are some examples:
[
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "STUDIED_AT",
        "tail": "Delhi Technological University",
        "tail_type": "College"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "PURSUING_COURSE",
        "tail": "Bachelor of Technology in software",
        "tail_type": "Course"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "82% in Class-XII",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nEducation\nDelhi Technological University August 2020 - June 2024\nBachelor of Technology in software(CGPA of 8.07) New Delhi, India\nKendriya Vidyalaya-Keshav Puram 82%\nClass-XII New Delhi, India\nKendriya Vidyalaya-Keshav Puram 84%\nClass-X New Delhi, India",
        "head": "Kendriya Vidyalaya-Keshav Puram",
        "head_type": "School",
        "relation": "SCORED",
        "tail": "84% in Class-X",
        "tail_type": "Grade"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "akash kumar prasad",
        "head_type": "Person",
        "relation": "WORKED_AS",
        "tail": "Software Developer Intern",
        "tail_type": "Role"
    },
    {
        "text": "akash kumar prasad\nNew Delhi, India\n♂phone+91 9205520953 /envel⌢peaakashkumarprasad se20b16 73@dtu.ac.in /linkedinLinkedIn ὋCPortfolio\nExperience\nSoftware Developer Intern (March 2024 - April 2024)\nAqeeq Technologies Remote, India\n•Worked directly with the co-founder and rest of the team, contributing to a 12% increase in project efficiency.\n•Support technology initiatives within the team. Contributed to the engineering culture that focuses on simple, intuitive,\nhigh-impact experiences. Took end-to-end ownership and responsibility across the full development lifecycle.",
        "head": "Aqeeq Technologies",
        "head_type": "Company",
        "relation": "EMPLOYED_BY",
        "tail": "akash kumar prasad",
        "tail_type": "Person"
    }
]
"""),("human","text: {{input_text}}")])


# llm_transformer = LLMGraphTransformer(
#     llm=llm_mistral,
#     node_properties=["description"],
#     relationship_properties=["description"],
#     allowed_nodes=["person name","college/organisation","course"],
#     prompt=template)
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
llm_openai = ChatOpenAI(model="gpt-4")

PROMPT = ChatPromptTemplate([('system',"""Analyze the following documents and list the different entities you find. Specifically, look for: 
Person names (e.g., Akansha, Aditi)
Programming language skills (e.g., Python, Java)
Course names (e.g., BTech, Phd)
Institute or college names (e.g., Indian Institute of Information Technology, Delhi)
Provide the entities in a clear and organized format.

-Steps-
1. Identify all relevant entities from the resume. For each identified entity, extract the following information:
- entity_name: Name of the person, skills, or course, capitalized
- entity_type: One of the following types: [Person, Skill, Course]
- entity_description: Comprehensive description of the entity's attributes and activities (for example, for a skill, mention the context or level of proficiency; for a course, mention the institution or time of completion)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why the source entity and the target entity are related (e.g., a skill that was acquired through a course)
- relationship_strength: an integer score between 1 to 10, indicating the strength of the relationship between the source entity and target entity

3. Return output in graph document as a single list of all the entities and relationships identified in steps 1 and 2. 
4. If you have to translate into graph document, just translate the descriptions, nothing else!


]"""),('human',"""entity_types: [Person, Skill, Course, Institute/College name], 
text: {{input_text}}""")])

llm_transformer = LLMGraphTransformer(
    llm=llm_openai,
    node_properties=["description"],
    relationship_properties=["description"],
    allowed_nodes=["Person","Skills","Institute","Degree","Projects","Job titles","Company names","Work experience","Certifications"])


def create_chunks(pdf_folder: str) -> pd.DataFrame:
    """Loads text from all PDF files in the specified folder and returns a DataFrame."""

    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    count =0

    for filename in os.listdir(pdf_folder):
        if count == 100:
            break
        count+=1
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(file_path)
            pages = []

            for page in loader.load():
                pages.append(page.page_content)

            combined_text = " ".join(pages)

            chunks = text_splitter.create_documents(
                [combined_text],
                metadatas=[{"name": filename}]
            )

            all_chunks.extend(chunks)

    return all_chunks

pdf_folder = "resume"

def save(data, filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            existing_data = pickle.load(f)
        existing_data.extend(data)
        with open(filename, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

def show_graph(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def process_text() -> List[GraphDocument]:
  doc_list = create_chunks(pdf_folder)
  graph_documents = []
  for i in doc_list:
    print(i)
    result = llm_transformer.convert_to_graph_documents([i])
    graph_documents.extend(result)
    print(result)
    save(result, 'graph_documents.pkl')
  return graph_documents


# create graph documents
# graph_documents = process_text()

graph_documents = show_graph("graph_documents.pkl")

graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)
# res = show_graph("graph_documents.pkl")
# print(len(res))





