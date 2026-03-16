import pytest
from langchain_community.utilities import requests
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy
from ragas.metrics.collections import FactualCorrectness

from utils.utils import get_llm_response, load_test_data


@pytest.mark.parametrize("get_data", load_test_data("test_factual_correctness.py"), indirect=True)
@pytest.mark.asyncio
async def test_relevance_factual(llm_wrapper, get_data):
    metrics = [ResponseRelevancy(llm=llm_wrapper),
               FactualCorrectness(llm=llm_wrapper)]

    eval_dataset = EvaluationDataset([get_data])

    # if we nor include the metrics , the ragas library will evaluate all available for LLM evaluation
    # results = evaluate(dataset=eval_dataset, metrics=metrics)
    results = evaluate(dataset=eval_dataset)
    print(results)
    for result in results:
        assert result["answer_relevancy"] > 0.8
    results.upload()


@pytest.fixture
def get_data(request):
    test_data = requests.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        reference=test_data["reference"]
    )
    return sample