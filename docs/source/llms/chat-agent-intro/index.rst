Tutorial: Getting Started with ChatAgent
========================================

Use when you want to develop and deploy a conversational agent that returns multiple messages, enabling returning intermediate steps for tool calling, tool call confirmation, and multi-agent support. Even if you're only doing a simple ChatModel, you should use ChatAgent upfront because it will give you greater flexibility to make your model more agentic.


Authoring the ChatAgent

Logging the ChatAgent 
- input example
- signature is set for you
- task type is set for you



What You'll Learn
-----------------

This guide will take you through the basics of using the ChatModel API to define custom conversational AI models. In particular, you will learn: 

#. How to map your application logic to the ``ChatModel``'s input/output schema
#. How to use the pre-defined inference parameters supported by ChatModels
#. How to pass custom parameters to a ChatModel using ``custom_inputs``
#. How :py:class:`~mlflow.pyfunc.ChatModel` compares to :py:class:`~mlflow.pyfunc.PythonModel` for defining custom chat models

# TODO: How to go from ChatModel to ChatAgent
---------------------------------------------------

# TODO: Open source LangGraph ChatAgent Example
---------------------------------------------------
