# This is the example chain as code that are logged as code using
# the mlflow.langchain.log_model API.
# The chain_as_code_driver.py is the driver code that logs the chain as code
# and loads the chain back using mlflow.langchain.load_model.
import os
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAI

import mlflow

mlflow.langchain.autolog()

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


prompt = PromptTemplate(
    template="You are a hello world bot.  Respond with a reply to the user's question that is fun and interesting to the user.  User's question: {question}",
    input_variables=["question"],
)

model = OpenAI(temperature=0.9)

chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | StrOutputParser()
)

question = {
    "messages": [
        {
            "role": "user",
            "content": "what is rag?",
        },
    ]
}

chain.invoke(question)

# Using mlflow.models.set_model, set the model that will loaded back
# using mlflow.langchain.load_model, eventually used for inference.
mlflow.models.set_model(model=chain)
