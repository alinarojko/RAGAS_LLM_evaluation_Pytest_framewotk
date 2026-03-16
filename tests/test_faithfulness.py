import pytest
import requests
from ragas import SingleTurnSample
from ragas.metrics.collections import Faithfulness

from utils.utils import load_test_data, get_llm_response


@pytest.mark.asyncio
@pytest.mark.parametrize("get_data", load_test_data("test_faithfulness.py"), indirect=True)
async def test_faithfulness(llm_wrapper, get_data):
    faithful = Faithfulness(llm=llm_wrapper)
    score = await faithful.single_turn_ascore(get_data)
    print(score)
    assert score > 0.8


@pytest.fixture
def get_data(request):
    test_data = requests.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")]
    )
    return sample
