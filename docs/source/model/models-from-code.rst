Models From Code Guide
======================

.. attention::
    Models from Code is available in MLflow 2.12.2 and above. If you are using a version earlier than what supports this feature, 
    you are required to use the legacy serialization methods outlined in the `Custom Python Model <../models.html#custom-python-models>`_ documentation.

.. note::
    Models from code is only available for `LangChain <../llms/langchain/index.html>`_ and custom ``pyfunc`` (PythonModel instances) models. If you are 
    using other libraries directly, using the provided saving and logging functionality within specific model flavors is recommended.


The models from code feature is a comprehensive overhaul of the process of defining, storing, and loading custom models. The key difference between 
legacy serialization and the models from code approach is in what is defined in the ``python_model`` argument within :py:func:`mlflow.pyfunc.log_model`. 

The greatest gain associated with using models from code for custom ``pyfunc`` implementations is in the reduction of repetitive trial-and-error debugging 
that can occur when working on an implementation. The workflow shown below illustrates how these two methdologies compare when working on a solution:

.. figure:: ../_static/images/models/models_from_code_journey.png
    :alt: Models from code comparison with legacy serialization
    :width: 80%
    :align: center

Differences with Legacy PythonModel logging
-------------------------------------------

In the legacy mode, an instance of your subclassed :py:class:`mlflow.pyfunc.PythonModel` is submitted in the call to ``log_model``. When called via an object
reference, MLflow will utilize ``cloudpickle`` to attempt to serialize your object. 

In models from code, instead of passing an object reference to an instance of your custom model, you will simply pass a path reference to a script that 
contains your model definition. When this mode is employed, MLflow will simply execute this script (along with any ``code_paths`` dependencies prior to running 
the main script) in the execution environment and instantiating whichever object you define in the call to :py:func:`mlflow.models.set_model`, assigning that 
object as the inference target. 

At no point in this process are there dependencies on serialization libraries such as `pickle <https://docs.python.org/3/library/pickle.html>`_ or 
`cloudpickle <https://pypi.org/project/cloudpickle/1.1.1/>`_, removing the broad limitations that these serialization packages have, such as:

- **Portability and Compatiblility**: Loading a pickle or cloudpickle file in a Python version that was different than the one used to serialize the object does not guarantee compatiblity.
- **Complex Object Serialization**: File handles, sockets, external connections, dynamic references, lambda functions and system resources are unavailable for pickling.
- **Readability**: Pickle and CloudPickle both store their serialized objects in a binary format that is impossible to read by humans.
- **Performance**: Object serialization and dependency inspection can be very slow, particularly for complex implementations with many code reference dependencies.

Core requirements for using Models From Code
--------------------------------------------

There are a few considerations to be aware of when evaluating whether your defined model is suitable for using the models from code approach for logging. 

- **Imports**: Models from code does not capture external references for non-pip installable packages. If you have external references (see the examples below), you must define these dependencies via ``code_paths`` arguments.
- **Execution during logging**: In order to validate that the script file that you're logging is valid, the code will be executed before being written to disk. 
- **Requirements inference**: Packages that are imported at the top of your defined model script will be inferred as requirements if they are installable from PyPI. 

.. tip::
    If you define import statements that are never used within your script, these will still be included in the requirements listing. It is recommended to use a linter
    that is capable of determining unused import statements while writing your implementation so that you are not including irrelevant package dependencies.

Using Models From Code in a Jupyter Notebook
--------------------------------------------

`Jupyter <https://jupyter.org/>`_ (IPython Notebooks) are a very convenient way to work with AI applications and modeling in general. One slight limitation that they 
have is in their cell-based execution model. Due to the nature of how they are defined and run, the models from code feature does not directly support defining 
a notebook as a model. Rather, this feature requires that models are defined as Python scripts (the file extension **must end in '.py'**). 

Fortunately, the folks that maintain the core kernel that Jupyter uses (`IPython <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`_) have created a 
number of magic commands that are usable within notebooks to enhance the usability of notebooks as a development environment for AI practitioners. One of the most 
useful magic commands that can be used within any notebook environment that is based upon IPython (``Jupyter``, ``Databricks Notebooks``, etc.) is the ``%%writefile`` command.

The `%%writefile <https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-writefile>`_ magic command, when written as the first line of a notebook 
cell, will capture the contents of the cell (not the entire notebook, mind you, only the current cell scope) with the exception of the magic command itself and write 
those contents to the file that you define. 

For example, running the following in a notebook:

.. code-block:: none

    %%writefile "./hello.py"

    print("hello!")

Will result in a file being created, located in the same directory as your notebook, that contains:

.. code-block:: python

    print("hello!")


.. note::
    There is an optional ``-a`` append command that can be used with the ``%%writefile`` magic command. This option will **append** the cell contents to the file 
    being targeted for saving the cell contents to. It is **not recommended** to use this option due to the chances of creating difficult-to-debug overrides within 
    a script that could contain multiple copies of your model definition logic. It is recommended to use the default behavior of ``%%writefile``, which is to overwrite 
    the local file each time that the cell is executed to ensure that the state of your cell's contents are always reflected in the saved script file.


Examples of Using Models From Code
----------------------------------
Each of these examples will show usage of the ``%%writefile`` magic command at the top of the script definition cell block in order to simulate defining the model code or other 
dependencies from within a single notebook. If you are writing your implementations within an IDE or a text editor, do not place this magic command at the top of your 
script.

.. tabs::

    .. tab:: Simple Example
        In this example, we will define a very basic  model that, when called via ``predict()``, will utilize the input float value as an exponent to the number ``2``.
        The first code block, repesenting a discrete notebook cell, will create a file named ``basic.py`` in the same directory as the notebook. The contents of this 
        file will be the model definition ``BasicModel``, as well as the import statements and the MLflow function ``set_model`` that will instantiate an instance of 
        this model to be used for inference.

        .. code-block:: python

            # If running in a Jupyter or Databricks notebook cell, uncomment the following line:
            # %%writefile "./basic.py"

            import pandas as pd
            from typing import List, Dict
            from mlflow.pyfunc import PythonModel
            from mlflow.models import set_model


            class BasicModel(PythonModel):
                def exponential(self, numbers):
                    return {f"{x}": 2**x for x in numbers}

                def predict(self, context, model_input) -> Dict[str, float]:
                    if isinstance(model_input, pd.DataFrame):
                        model_input = model_input.to_dict()[0].values()
                    return self.exponential(model_input)


            # Specify which definition in this script represents the model instance
            set_model(BasicModel())

        The next section shows another cell that contains the logging logic. 

        .. code-block:: python

            import mlflow

            mlflow.set_experiment("Basic Model From Code")

            model_path = "basic.py"

            with mlflow.start_run():
                model_info = mlflow.pyfunc.log_model(
                    python_model=model_path,  # Define the model as the path to the script that was just saved
                    artifact_path="arithemtic_model",
                    input_example=[42.0, 24.0],
                )


        Looking at this stored model within the MLflow UI, we can see that the script in the first cell was recorded as an artifact to the run. 
        
        .. figure:: ../_static/images/models/basic_model_from_code_ui.png
            :alt: The MLflow UI showing the stored model code as a serialized python script
            :width: 80%
            :align: center

        When we load this model via ``mlflow.pyfunc.load_model()``, this script will be executed and an instance of ``BasicModel`` will be constructed, exposing the ``predict`` 
        method as our entry point for inference, just as with the alternative legacy mode of logging a custom model.

        .. code-block:: python
            
            my_model = mlflow.pyfunc.load_model(model_info.model_uri)
            my_model.predict([2.2, 3.1, 4.7])

            # or, with a Pandas DataFrame input
            my_model.predict(pd.DataFrame([5.0, 6.0, 7.0]))
    
    .. tab:: Models with Code Paths dependencies

        In this example, we will explore a more complex scenario that demonstrates how to work with multiple Python scripts and leverage the ``code_paths`` 
        feature in MLflow for model management. Specifically, we will define a ``Calculator`` class in one script, which performs basic arithmetic 
        operations, and then use this class within an ``ArithmeticModel`` custom ``PythonModel`` that we will define in a separate script. 
        This model will be logged with MLflow, allowing us to perform predictions using the stored model.

        This tutorial will show you how to:

        - Create multiple Python files from within a Jupyter notebook.
        - Log a custom model with MLflow that relies on external code defined in another file.
        - Use the ``code_paths`` feature to include additional scripts when logging the model, ensuring that all dependencies are available when the model is loaded for inference.

        In the first step, we define a ``Calculator`` class in a file named ``calculator.py``. This class includes a basic arithmetic operation, such as 
        adding two numbers, and also keeps a history of operations performed. The purpose of this class is to encapsulate the logic that will be used 
        later in the MLflow model.

        The following code block writes the ``Calculator`` class definition to ``calculator.py``:
        
        .. code-block:: python

            # If running in a Jupyter or Databricks notebook cell, uncomment the following line:
            # %%writefile "./calculator.py"

            from typing import List, TypeVar


            T = TypeVar("T", int, float, complex)


            class Calculator:
                def __init__(self):
                    self.history = []

                def add(self, a: T, b: T) -> T:
                    result = a + b
                    self.history.append(f"The sum of {a} and {b} is {result}")
                    return {"result": result, "history": self.history}

        This script defines a versatile calculator that can handle different types of numerical inputs (int, float, complex). The ``add`` method not only 
        computes the sum of two numbers but also records the operation in a history log. This history can be useful for debugging or tracking the 
        sequence of operations performed by the model.

        Next, we create a new file, ``math_model.py``, which contains the ``ArithmeticModel`` class. This class will be responsible for loading the ``Calculator`` class, 
        performing predictions, and validating the input data types. The predict method will leverage the ``Calculator`` class to perform the addition of two numbers provided as input.

        The ``load_context`` method within ``ArithmeticModel`` ensures that the ``Calculator`` class, defined in the external ``calculator.py`` script, is loaded 
        and available for use when the model is deployed or invoked.

        The following code block writes the ``ArithmeticModel`` class definition to ``math_model.py``:

        .. code-block:: python

            # If running in a Jupyter or Databricks notebook cell, uncomment the following line:
            # %%writefile "./math_model.py"

            from typing import Dict, Any, Union
            from mlflow.pyfunc import PythonModel
            from mlflow.models import set_model


            class ArithmeticModel(PythonModel):
                def __init__(self):
                    self.model = None
                    self.types = (int, float, complex)

                def load_context(self, context):
                    # We are loading from an external module that is defined within a code_paths path
                    from calculator import Calculator

                    self.model = Calculator()

                def predict(
                    self, context, model_input: Dict[str, Any], params=None
                ) -> Union[int, float, complex]:
                    try:
                        a = model_input["a"]
                        b = model_input["b"]
                    except KeyError as e:
                        raise KeyError(f"Missing required input: {e.args[0]}") from e

                    if not isinstance(a, self.types) or not isinstance(b, self.types):
                        raise ValueError(
                            f"Input types must be one of {self.types}, but received: {type(a)}, {type(b)}"
                        )

                    return self.model.add(a, b)


            set_model(ArithmeticModel())

        This model introduces error handling by checking the existence and types of the inputs, ensuring robustness. It serves as a practical example of 
        how custom logic can be encapsulated within an MLflow model while leveraging external dependencies.

        Once the ``ArithmeticModel`` is defined, we can proceed to log it with MLflow. This process involves specifying the path to the ``math_model.py`` 
        script and using the ``code_paths`` parameter to include ``calculator.py`` as a dependency. This ensures that when the model is loaded in 
        a different environment or on another machine, all necessary code files are available for proper execution.

        The following code block demonstrates how to log the model using MLflow:

        .. code-block:: python

            import mlflow

            mlflow.set_experiment("Arithemtic Model From Code")

            model_path = "math_model.py"

            with mlflow.start_run():
                model_info = mlflow.pyfunc.log_model(
                    python_model=model_path,  # The model is defined as the path to the script containing the model definition
                    artifact_path="arithemtic_model",
                    code_paths=[
                        "calculator.py"
                    ],  # dependency definition included for the model to successfully import the implementation
                )

        This step registers the ``ArithmeticModel`` with MLflow, ensuring that both the primary model script and its dependencies are stored as 
        artifacts. By including ``calculator.py`` in the ``code_paths`` argument, we ensure that the model can be reliably reloaded and used for 
        predictions, regardless of the environment in which it is deployed.

        After logging the model, it can be loaded back into the notebook or any other environment that has access to the MLflow tracking server. 
        When the model is loaded, the ``calculator.py`` script will be executed along with the ``math_model.py`` script, ensuring that the 
        ``Calculator`` class is available for use by the ``ArithmeticModel``.

        The following code block demonstrates how to load the model and make predictions:

        .. code-block:: python

            my_model_from_code = mlflow.pyfunc.load_model(model_info.model_uri)
            my_model_from_code.predict({"a": 42, "b": 9001})
            my_model_from_code.predict({"a": 37.25, "b": 5.32e7})

        This example showcases the model's ability to handle different numerical inputs, perform addition, and maintain a history of calculations. 
        The output of these predictions includes both the result of the arithmetic operation and the history log, which can be useful for auditing and 
        tracing the computations performed by the model.

        .. code-block:: text

            {
                'result': 53200037.25,
                'history': [
                    'The sum of 42 and 9001 is 9043',
                    'The sum of 37.25 and 53200000.0 is 53200037.25'
                ]
            }

        Looking at the stored model within the MLflow UI, you can see that both the ``math_model.py`` and ``calculator.py`` scripts are recorded as 
        artifacts in the run. This comprehensive logging allows you to track not just the model's parameters and metrics but also the code that 
        defines its behavior, making it visible and debuggable directly from within the UI.

        .. figure:: ../_static/images/models/model_from_code_code_paths.png
            :alt: The MLflow UI showing models from code usage along with dependent code_paths script stored in the model artifacts
            :width: 80%
            :align: center


    .. tab:: Models From Code with LangChain

        In this slightly more advanced example, we will explore how to use the `MLflow LangChain integration <../llms/langchain/index.html>`_ to define 
        and manage a chain of operations for an AI model. This chain will help generate landscape design recommendations based on specific regional 
        and area-based inputs. The example showcases how to define a custom prompt, use a large language model (LLM) for generating responses, and 
        log the entire setup as a model using MLflow's tracking features.

        This tutorial will guide you through:

        - Writing a script to define a custom LangChain model that processes input data to generate landscape design recommendations.
        - Logging the model with MLflow using the langchain integration, ensuring the entire chain of operations is captured.
        - Loading and using the logged model for making predictions in different contexts.


        First, we will create a Python script named ``mfc.py``, which defines the chain of operations for generating landscape design recommendations. 
        This script utilizes the LangChain library along with MLflow's ``autolog`` feature for enabling the `capture of traces <../llms/tracing/index.html>`_.

        In this script:

        - **Custom Functions** (get_region and get_area): These functions extract specific pieces of information (region and area) from the input data.
        - **Prompt Template**: A ``PromptTemplate`` is defined to structure the input for the language model, specifying the task and context in which the model will operate.
        - **Model Definition**: We use the ``ChatOpenAI`` model to generate responses based on the structured prompt.
        - **Chain Creation**: The chain is created by connecting the input processing, prompt template, model invocation, and output parsing steps.
        
        The following code block writes this chain definition to the mfc.py file:
        
        .. code-block:: python

            # If running in a Jupyter or Databricks notebook cell, uncomment the following line:
            # %%writefile "./mfc.py"

            import os
            from operator import itemgetter

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import RunnableLambda
            from langchain_openai import ChatOpenAI

            import mlflow

            mlflow.set_experiment("Homework Helper")

            mlflow.langchain.autolog()


            def get_region(input_data):
                default = "Virginia, USA"
                if isinstance(input_data[0], dict):
                    return input_data[0].get("content").get("region", default)
                return default


            def get_area(input_data):
                default = "5000 square feet"
                if isinstance(input_data[0], dict):
                    return input_data[0].get("content").get("area", default)
                return default


            prompt = PromptTemplate(
                template="You are a highly accomplished landscape designer that provides suggestions for landscape design decisions in a particular"
                " geographic region. Your goal is to suggest low-maintenance hardscape and landscape options that involve the use of materials and"
                " plants that are native to the region mentioned. As part of the recommendations, a general estimate for the job of creating the"
                " project should be provided based on the square footage estimate. The region is: {region} and the square footage estimate is:"
                " {area}. Recommendations should be for a moderately sophisticated suburban housing community within the region.",
                input_variables=["region", "area"],
            )

            model = ChatOpenAI(model="gpt-4o", temperature=0.95, max_tokens=4096)

            chain = (
                {
                    "region": itemgetter("messages") | RunnableLambda(get_region),
                    "area": itemgetter("messages") | RunnableLambda(get_area),
                }
                | prompt
                | model
                | StrOutputParser()
            )

            mlflow.models.set_model(chain)

        This script encapsulates the logic required to construct the full chain using the 
        `LangChain Expression Language (LCEL) <https://python.langchain.com/v0.1/docs/expression_language/>`_, as well as the custom default logic 
        that the chain will use for input processing. The defined chain is then specified as the model's interface object using the ``set_model`` function.

        Once the chain is defined in ``mfc.py``, we log it using MLflow. This step involves specifying the path to the script that contains the chain 
        definition and using MLflow's ``langchain`` integration to ensure that all aspects of the chain are captured.

        The ``input_example`` provided to the logging function serves as a template to demonstrate how the model should be invoked. This example is 
        also stored as part of the logged model, making it easier to understand and replicate the model's use case.

        The following code block demonstrates how to log the LangChain model using MLflow:

        .. code-block:: python

            import mlflow

            mlflow.set_experiment("Landscaping")

            chain_path = "./mfc.py"

            input_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "region": "Austin, TX, USA",
                            "area": "1750 square feet",
                        },
                    }
                ]
            }

            with mlflow.start_run():
                info = mlflow.langchain.log_model(
                    lc_model=chain_path,  # Defining the model as the script containing the chain definition and the set_model call
                    artifact_path="chain",
                    input_example=input_example,
                )

        In this step, the entire chain of operations, from input processing to AI model inference, is logged as a single, cohesive model. Avoiding the 
        potential complexities associated with object serialization of the defined chain components, using the models from code feature ensures that 
        the exact code and logic that were used to develop and test a chain is what is executed when deploying the application without the risk of 
        incomplete or non-existent serialization capabilities.

        After logging the model, it can be loaded back into your environment for inference. This step demonstrates how to load the chain and 
        use it to generate landscape design recommendations based on new input data.

        The following code block shows how to load the model and run predictions:

        .. code-block:: python

            # Load the model and run inference
            landscape_chain = mlflow.langchain.load_model(model_uri=info.model_uri)

            question = {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "region": "Raleigh, North Carolina USA",
                            "area": "3850 square feet",
                        },
                    },
                ]
            }

            response = landscape_chain.invoke(question)

        This code block demonstrates how to invoke the loaded chain with new data, generating a response that provides landscape design suggestions 
        tailored to the specified region and area. 

        Once the model is logged, you can explore its details in the MLflow UI. The interface will show the script ``mfc.py`` as an artifact of the 
        logged model, along with the chain definition and associated metadata. This allows you to easily review the model's components, 
        input examples, and other key information.

        .. figure:: ../_static/images/models/langchain_model_from_code.png
            :alt: The MLflow UI showing models from code usage and the mfc.py script that defines the LangChain LCEL chain definition
            :width: 80%
            :align: center

        When you load this model using :py:func:`mlflow.langchain.load_model`, the entire chain defined in ``mfc.py`` is executed, and the model
        behaves as expected, generating AI-driven recommendations for landscape design. 



Tips on Using Models From Code
------------------------------

Dependency Management
^^^^^^^^^^^^^^^^^^^^^

- **Import Statements**: Ensure that all necessary import statements are included in your model script. Missing imports will cause runtime errors when the model is executed. By specifying all dependencies explicitly, you make sure that the correct versions are captured in the ``requirements.txt`` file during the model logging process, ensuring compatibility in different environments.

- **Non-Pip Installable Dependencies**: If your model relies on custom or non-PyPI packages, make sure to include these via the ``code_paths`` argument when logging the model. This will package the additional scripts or modules with the model, allowing it to be used seamlessly in different environments.

Script Execution & Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Security**: Be **very careful** about what is defined in your script. If there are API Tokens, sensitive data, comments that contain authentication information, or anything else that you don't want to be visible in plain text are removed from your script. The models from code feature stores your notebook cell (when used with the magic ``%%writefile`` command) or script in plaintext. 

- **Pre-Execution**: Remember that the script you log will be **executed during the logging process**. This execution helps validate that the script is functional and that the defined model instance is properly instantiated. Ensure that any side effects of this execution are handled, such as temporary files or access to external resources.

Requirements Inference
^^^^^^^^^^^^^^^^^^^^^^
- **Automatic Detection**: MLflow will automatically detect and infer the requirements for packages that are importable via PyPI if they are present in the script. If certain packages are not necessary for the model, it's advisable to remove their import statements to avoid unnecessary dependencies being added to the environment when the model is loaded or deployed.

Managing Multiple Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^
- **Use of code_paths**: If your model depends on multiple scripts or modules, these should be included using the ``code_paths`` parameter when logging the model. This ensures that all dependencies are included and available when the model is loaded later.

Jupyter Notebook Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Notebook to Script Conversion**: Since Models from Code requires a ``.py`` script, you can use magic commands like ``%%writefile`` in Jupyter notebooks to save cells as Python scripts. This is especially useful when working in a notebook environment and allows you to convert your work into a format suitable for logging with MLflow.

- **Avoid Appending**: While the ``-a`` append option of ``%%writefile`` can be useful, it can also introduce hard-to-debug issues by merging multiple cells into one file. It’s generally safer to overwrite the file to reflect the latest state of your code.

Model Loading and Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Script-Based Model Loading**: When you load a model logged using Models from Code, the script is executed, and the defined model instance is created dynamically. Ensure that the model script is self-contained and does not rely on external state, as this could lead to inconsistent behavior during inference.

- **Input Examples**: Providing input examples during the logging process can help in understanding how the model is intended to be used. These examples are saved alongside the model and can serve as a reference or for testing purposes when the model is deployed.
