import os
import requests
import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference

from utils.utils import load_test_data, get_llm_response


@pytest.mark.asyncio
@pytest.mark.parametrize("get_data", load_test_data("test_context_recall.py"), indirect=True)
async def test_context_precision(llm_wrapper, get_data):
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
    score = await context_precision.single_turn_score(get_data)
    print(score)
    assert score > 0.8


@pytest.fixture
def get_data(request):
    test_data = requests.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=responseDict["retrieved_docs"][0]["page_content"]
    )
    return sample