.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - ChatAgent
     - ChatModel
   * - When to use
     -  Use when you want to develop and deploy a conversational agent that returns multiple messages, enabling returning intermediate steps for tool calling, tool call confirmation, and multi-agent support. Even if you're only doing a simple ChatModel, you should use ChatAgent upfront because it will give you greater flexibility to make your model more agentic.
   * - Interface
     - **Fixed** to a ChatAgent schema that is mostly OpenAI compatible. Includes additional fields like attachments to better support additional tool output.
     - **Fixed** to OpenAI's chat schema.
