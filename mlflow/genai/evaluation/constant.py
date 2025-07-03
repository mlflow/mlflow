class AgentEvaluationReserverKey:
    """
    Expectation column names that are used by Agent Evaluation.
    Ref: https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluation-schema
    """

    EXPECTED_RESPONSE = "expected_response"
    EXPECTED_RETRIEVED_CONTEXT = "expected_retrieved_context"
    EXPECTED_FACTS = "expected_facts"
    GUIDELINES = "guidelines"

    @classmethod
    def get_all(cls) -> set[str]:
        return {
            cls.EXPECTED_RESPONSE,
            cls.EXPECTED_RETRIEVED_CONTEXT,
            cls.EXPECTED_FACTS,
            cls.GUIDELINES,
        }


# A column name for storing custom expectations dictionary in Agent Evaluation.
AGENT_EVAL_CUSTOM_EXPECTATION_KEY = "custom_expected"
