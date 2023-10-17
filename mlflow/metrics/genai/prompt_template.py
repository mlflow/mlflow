import string
from typing import Any, List, Optional

from mlflow.exceptions import MlflowException


class PromptTemplate:
    """A prompt template for a language model.

    A prompt template consists of a string template. It accepts a set of parameters
    from the user that can be used to generate a prompt for a language model.

    The template can be formatted using f-strings.

    Example:

        .. code-block:: python

            from mlflow.metrics.genai.prompt_template import PromptTemplate

            # Instantiation using initializer
            prompt = PromptTemplate(template_str="Say {foo} {baz}", variables=["foo", "baz"])

            # Instantiation using partial_fill
            prompt = PromptTemplate(template_str="Say {foo} {baz}").partial_fill(foo="bar")

            # Format the prompt
            prompt.format(baz="qux")
    """

    def __init__(self, template_str: str, variables: Optional[List[str]] = None):
        self.template_str = template_str
        extracted_variables = [
            fname for _, fname, _, _ in string.Formatter().parse(template_str) if fname
        ]
        if variables:
            if not all(item in variables for item in extracted_variables):
                raise MlflowException(
                    f"The provided variables {variables} are not a subset of "
                    f"the extracted variables {extracted_variables} from the template string"
                )
            self.variables = variables
        else:
            # Automatically parse variables from template string
            self.variables = extracted_variables

    def format(self, **kwargs: Any) -> str:
        # Only keep the kwargs that are in the variables
        kwargs = {k: v for k, v in kwargs.items() if k in self.variables}

        # Format the prompt with the provided values
        return self.template_str.format(**kwargs)

    def partial_fill(self, **kwargs: Any) -> "PromptTemplate":
        # Create a safe dictionary that returns the key if it doesn't exist in the dictionary
        safe_dict = {k: kwargs.get(k, "{" + k + "}") for k in self.variables}

        # Fill in the provided values, and return a new PromptTemplate
        new_template_str = self.template_str.format_map(safe_dict)
        unfilled_variables = [var for var in self.variables if var not in kwargs.keys()]
        return PromptTemplate(template_str=new_template_str, variables=unfilled_variables)

    def __str__(self):
        return self.template_str
