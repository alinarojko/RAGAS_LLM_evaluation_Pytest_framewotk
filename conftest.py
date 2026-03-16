import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

os.environ["OPENAI_API_KEY"] = "my_api_key_for_open_ai"
os.environ["RAGAS_APP_TOKEN"] = "my_ragas_token_to see results on the dashboard"


@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperatura=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm