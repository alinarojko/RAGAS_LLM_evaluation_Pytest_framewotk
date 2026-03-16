import pytest
import requests
from ragas import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics._topic_adherence import TopicAdherenceScore
from ragas.metrics.collections import Faithfulness

from utils.utils import load_test_data, get_llm_response


@pytest.mark.asyncio
# @pytest.mark.parametrize("get_data", load_test_data("test_faithfulness.py"), indirect=True)
async def test_topic_adherence(llm_wrapper, get_data):
    topic_adherence = TopicAdherenceScore(llm=llm_wrapper)
    score = await topic_adherence.multi_turn_ascore(get_data)
    print(score)
    assert score > 0.8


@pytest.fixture
def get_data(request):
    test_data = requests.param
    responseDict = get_llm_response(test_data)

    conversation = [
        HumanMessage(content="Question1 from the user"),
        AIMessage(content="The response form the AI"),
        HumanMessage(content="Question2 from the user, about the context"),
        AIMessage(content="The response form the AI to the second message")
    ]

    references = [
        """
        The AI should:
        1.Give results related to the user question.
        2. The responce to the second question should be relevant to the first response"""
    ]

    sample = MultiTurnSample(user_input=conversation, reference_topic=references)
    return sample
