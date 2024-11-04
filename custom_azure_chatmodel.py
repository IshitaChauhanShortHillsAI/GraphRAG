from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import os
from langchain_mistralai import ChatMistralAI

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_TYPE=os.getenv('OPENAI_API_TYPE')
OPENAI_API_VERSION=os.getenv('OPENAI_API_VERSION')
MODEL_NAME=os.getenv('MODEL_NAME')
DEPLOYMENT_NAME=os.getenv('DEPLOYMENT_NAME')
USER_ID=os.getenv('USER_ID')

DEPLOYMENT_NAME4=os.getenv('DEPLOYMENT_NAME4')
MODEL_NAME4=os.getenv('MODEL_NAME4')
OPENAI_API_KEY4=os.getenv('OPENAI_API_KEY4')
OPENAI_API_BASE4=os.getenv('OPENAI_API_BASE4')
USER_ID4=os.getenv('USER_ID4')

llm = AzureChatOpenAI(azure_endpoint= OPENAI_API_BASE4,
            default_headers={"User-Id": USER_ID4},
            api_key = OPENAI_API_KEY4,
            temperature=0.1,
            deployment_name=DEPLOYMENT_NAME4,
            model_name=MODEL_NAME4,
            api_version=OPENAI_API_VERSION)

os.environ['MISTRAL_API_KEY']=os.getenv('MISTRAL_API_KEY')

llm_mistral = ChatMistralAI(model="mistral-large-latest")

from typing import Optional

class Joke(BaseModel):
    '''Joke to tell user.'''

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


structured_llm = llm_mistral.with_structured_output(Joke)
res3 = structured_llm.invoke("Tell me a joke about cats")

structured_llm = llm.with_structured_output(Joke)
res2 = structured_llm.invoke("Tell me a joke about cats")

os.environ['OPENAI_API_KEY']=os.environ['OPENAI_API_KEY']

llm_openai = ChatOpenAI(model="gpt-4")
structured_llm = llm_openai.with_structured_output(Joke)
res = structured_llm.invoke("Tell me a joke about cats")
print(res)
# print(res3,"/n",res2)