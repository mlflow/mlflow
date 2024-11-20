.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - ChatModel
     - PythonModel
   * - When to use
     - Use when you want to develop and deploy a conversational model with **standard** chat schema compatible with OpenAI spec.
     - Use when you want **full control** over the model's interface or customize every aspect of your model's behavior.
   * - Interface
     - **Fixed** to OpenAI's chat schema.
     - **Full control** over the model's input and output schema.
   * - Setup
     - **Quick**. Works out of the box for conversational applications, with pre-defined model signature and input example.
     - **Custom**. You need to define model signature or input example yourself.
   * - Complexity
     - **Low**. Standardized interface simplified model deployment and integration.
     - **High**. Deploying and integrating the custom PythonModel may not be straightforward. E.g., The model needs to handle Pandas DataFrames as MLflow converts input data to DataFrames before passing it to PythonModel.
