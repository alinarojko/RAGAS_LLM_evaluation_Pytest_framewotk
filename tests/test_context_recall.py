import os
import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._context_recall import LLMContextRecall

from utils.utils import get_llm_response, load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize("get_data", load_test_data("test_context_recall.py"), indirect=True)
async def test_context_recall(llm_wrapper, get_data):
    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_score(get_data)
    print(score)
    assert score > 0.8


@pytest.fixture
def get_data(request):
    test_data = requests.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_context=responseDict["answer"],
        reference=test_data["reference"]
    )
    return sample
