from langgraph.graph.graph import CompiledGraph

import mlflow


def test_langgraph_save_as_code():
    input_example = {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model="tests/langchain/sample_code/langgraph.py",
            artifact_path="langgraph",
            input_example=input_example,
        )

    loaded_graph = mlflow.langchain.load_model(model_info.model_uri)
    assert isinstance(loaded_graph, CompiledGraph)
    response = loaded_graph.invoke(input_example)
    # The OpenAI mock should return the same response as input
    assert response["messages"][0].content == "what is the weather in sf?"

    for chunk in loaded_graph.stream(input_example):
        assert (
            chunk["agent"]["messages"][0].content
            == '[{"role": "user", "content": "what is the weather in sf?"}]'
        )

    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    response = loaded_pyfunc.predict(input_example)[0]
    assert response["messages"][0].content == "what is the weather in sf?"

    for chunk in loaded_pyfunc.predict_stream(input_example, params={"stream_mode": "values"}):
        assert (
            chunk["agent"]["messages"][0].content
            == '[{"role": "user", "content": "what is the weather in sf?"}]'
        )
