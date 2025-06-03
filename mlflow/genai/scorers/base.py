import functools
import inspect
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, PrivateAttr

from mlflow.entities import Assessment, Feedback
from mlflow.entities.assessment import DEFAULT_FEEDBACK_NAME
from mlflow.entities.trace import Trace
from mlflow.utils.annotations import experimental


@experimental
class Scorer(BaseModel):
    name: str
    aggregations: Optional[list] = None

    # Private attributes to store source code
    _run_source: Optional[str] = PrivateAttr(default=None)
    _call_source: Optional[str] = PrivateAttr(default=None)
    _call_signature: Optional[str] = PrivateAttr(default=None)
    _original_func_name: Optional[str] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Extract and store source code after initialization."""
        super().model_post_init(__context)

        # Skip source code extraction for builtin scorers
        # Check if the instance is a subclass of builtin scorer (without importing it to avoid circular imports)
        if hasattr(self.__class__, "__module__") and "builtin_scorers" in self.__class__.__module__:
            return

        self._extract_source_code()

    def _extract_source_code(self):
        """Extract source code for run and the original decorated function."""
        from mlflow.genai.scorers.utils import extract_function_body

        # Extract run method source (always the same base implementation)
        try:
            run_body, _ = extract_function_body(self.run)
            object.__setattr__(self, "_run_source", run_body)
        except Exception:
            try:
                run_source = inspect.getsource(self.run)
                object.__setattr__(self, "_run_source", run_source)
            except Exception:
                # TODO: print a warning
                object.__setattr__(self, "_run_source", None)

        # Extract original function source (for decorator-created scorers)
        if hasattr(self, "_original_func") and self._original_func:
            try:
                call_body, _ = extract_function_body(self._original_func)
                object.__setattr__(self, "_call_source", call_body)
                object.__setattr__(self, "_original_func_name", self._original_func.__name__)
            except Exception:
                try:
                    call_source = inspect.getsource(self._original_func)
                    object.__setattr__(self, "_call_source", call_source)
                    object.__setattr__(self, "_original_func_name", self._original_func.__name__)
                except Exception:
                    # TODO: print a warning
                    object.__setattr__(self, "_call_source", None)
                    object.__setattr__(self, "_original_func_name", None)

            # Store the signature of the original function
            try:
                signature = str(inspect.signature(self._original_func))
                object.__setattr__(self, "_call_signature", signature)
            except Exception:
                object.__setattr__(self, "_call_signature", None)

    def model_dump(self, **kwargs) -> dict:
        """Override model_dump to include source code."""
        data = super().model_dump(**kwargs)

        # For builtin scorers, store the class information instead of source code
        # We detect builtin scorers by checking if their module contains 'builtin_scorers'
        # (e.g., 'mlflow.genai.scorers.builtin_scorers'). This approach allows us to use a simpler serialization
        # strategy: store class name + parameters instead of extracting and storing complex source code.
        if hasattr(self.__class__, "__module__") and "builtin_scorers" in self.__class__.__module__:
            data["_builtin_scorer_class"] = self.__class__.__name__
            # Also include any additional fields like required_columns
            if hasattr(self, "required_columns"):
                data["required_columns"] = self.required_columns
        elif hasattr(self, "_original_func") and self._original_func:
            # Add source code to the serialized data if available (for decorator-created scorers)
            if hasattr(self, "_run_source") and self._run_source:
                data["run_source"] = self._run_source
            if hasattr(self, "_call_source") and self._call_source:
                data["__call___source"] = self._call_source
            if hasattr(self, "_call_signature") and self._call_signature:
                data["__call___signature"] = self._call_signature
            if hasattr(self, "_original_func_name") and self._original_func_name:
                data["original_func_name"] = self._original_func_name
        else:
            # This is neither a builtin scorer nor a decorator scorer
            # Check if it's an unsupported direct subclass of Scorer
            base_call_method = Scorer.__call__
            current_call_method = self.__class__.__call__

            # If the __call__ method has been overridden (not the base NotImplementedError one)
            # then this is likely a direct subclass, which we don't support for serialization
            if current_call_method is not base_call_method:
                raise ValueError(
                    f"Unsupported scorer type: {self.__class__.__name__}. "
                    f"Scorer serialization only supports:\n"
                    f"1. Builtin scorers (from mlflow.genai.scorers.builtin_scorers)\n"
                    f"2. Decorator-created scorers (using @scorer decorator)\n"
                    f"Direct subclassing of Scorer is not supported for serialization. "
                    f"Please use the @scorer decorator instead."
                )

        return data

    @classmethod
    def model_validate(cls, obj: Any) -> "Scorer":
        """Override model_validate to reconstruct scorer from source code."""
        if isinstance(obj, dict):
            # Check if this is a builtin scorer
            builtin_class_name = obj.pop("_builtin_scorer_class", None)
            if builtin_class_name:
                # Import and reconstruct the builtin scorer
                from mlflow.genai.scorers import builtin_scorers

                scorer_class = getattr(builtin_scorers, builtin_class_name)

                # Get the valid field names for this scorer class from its model fields
                valid_fields = set(scorer_class.model_fields.keys())

                # Create instance with all the preserved data that matches valid fields
                constructor_args = {k: v for k, v in obj.items() if k in valid_fields}
                return scorer_class(**constructor_args)

            # Extract source code fields for decorator-created scorers
            run_source = obj.pop("run_source", None)
            call_source = obj.pop("__call___source", None)
            call_signature = obj.pop("__call___signature", None)
            original_func_name = obj.pop("original_func_name", None)

            # If we have the original function source, recreate the scorer using the decorator
            if call_source and call_signature and original_func_name:
                # Recreate the original function
                recreated_func = cls._recreate_function(
                    call_source, call_signature, original_func_name
                )

                if recreated_func:
                    # Apply the scorer decorator to recreate the scorer
                    recreated_scorer = scorer(
                        recreated_func, name=obj.get("name"), aggregations=obj.get("aggregations")
                    )
                    return recreated_scorer
                else:
                    raise ValueError("Failed to recreate function from source code. ")

            # If we reach here, the serialized data is invalid
            raise ValueError(
                "Invalid serialized scorer data. Expected either '_builtin_scorer_class' "
                "or source code fields ('__call___source', '__call___signature', 'original_func_name')."
            )

        return super().model_validate(obj)

    @classmethod
    def _recreate_function(cls, source: str, signature: str, func_name: str) -> Optional[Callable]:
        """Recreate a function from its source code."""
        try:
            # Parse the signature to build the function definition
            import re

            sig_match = re.match(r"\((.*?)\)", signature)
            if not sig_match:
                return None

            params_str = sig_match.group(1).strip()

            # Build the function definition
            func_def = f"def {func_name}({params_str}):\n"
            # Indent the source code
            indented_source = "\n".join(f"    {line}" for line in source.split("\n"))
            func_def += indented_source

            local_namespace = {}

            # Execute the function definition in the local namespace
            exec(func_def, globals(), local_namespace)

            # Return the recreated function
            return local_namespace[func_name]

        except Exception:
            return None

    def run(self, *, inputs=None, outputs=None, expectations=None, trace=None):
        from mlflow.evaluation import Assessment as LegacyAssessment

        merged = {
            "inputs": inputs,
            "outputs": outputs,
            "expectations": expectations,
            "trace": trace,
        }
        # Filter to only the parameters the function actually expects
        sig = inspect.signature(self.__call__)
        filtered = {k: v for k, v in merged.items() if k in sig.parameters}
        result = self(**filtered)
        if not (
            # TODO: Replace 'Assessment' with 'Feedback' once we migrate from the agent eval harness
            isinstance(result, (int, float, bool, str, Assessment, LegacyAssessment))
            or (
                isinstance(result, list)
                and all(isinstance(item, (Assessment, LegacyAssessment)) for item in result)
            )
        ):
            if isinstance(result, list) and len(result) > 0:
                result_type = "list[" + type(result[0]).__name__ + "]"
            else:
                result_type = type(result).__name__
            raise ValueError(
                f"{self.name} must return one of int, float, bool, str, "
                f"Feedback, or list[Feedback]. Got {result_type}"
            )

        if isinstance(result, Feedback) and result.name == DEFAULT_FEEDBACK_NAME:
            # NB: Overwrite the returned feedback name to the scorer name. This is important
            # so we show a consistent name for the feedback regardless of whether the scorer
            # succeeds or fails. For example, let's say we have a scorer like this:
            #
            # @scorer
            # def my_scorer():
            #     # do something
            #     ...
            #     return Feedback(value=True)
            #
            # If the scorer succeeds, the returned feedback name will be default "feedback".
            # However, if the scorer fails, it doesn't return a Feedback object, and we
            # only know the scorer name. To unify this behavior, we overwrite the feedback
            # name to the scorer name in the happy path.
            # This will not apply when the scorer returns a list of Feedback objects.
            # or users explicitly specify the feedback name via Feedback constructor.
            result.name = self.name

        return result

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: Optional[dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> Union[int, float, bool, str, Feedback, list[Feedback]]:
        """
        Implement the custom scorer's logic here.


        The scorer will be called for each row in the input evaluation dataset.

        Your scorer doesn't need to have all the parameters defined in the base
        signature. You can define a custom scorer with only the parameters you need.
        See the parameter details below for what values are passed for each parameter.

        .. list-table::
            :widths: 20 20 20
            :header-rows: 1

            * - Parameter
              - Description
              - Source

            * - ``inputs``
              - A single input to the target model/app.
              - Derived from either dataset or trace.

                * When the dataset contains ``inputs`` column, the value will be
                  passed as is.
                * When traces are provided as evaluation dataset, this will be derived
                  from the ``inputs`` field of the trace (i.e. inputs captured as the
                  root span of the trace).

            * - ``outputs``
              - A single output from the target model/app.
              - Derived from either dataset, trace, or output of ``predict_fn``.

                * When the dataset contains ``outputs`` column, the value will be
                  passed as is.
                * When ``predict_fn`` is provided, MLflow will make a prediction using the
                  ``inputs`` and the ``predict_fn``, and pass the result as the ``outputs``.
                * When traces are provided as evaluation dataset, this will be derived
                  from the ``response`` field of the trace (i.e. outputs captured as the
                  root span of the trace).

            * - ``expectations``
              - Ground truth or any expectation for each prediction, e.g. expected retrieved docs.
              - Derived from either dataset or trace.

                * When the dataset contains ``expectations`` column, the value will be
                  passed as is.
                * When traces are provided as evaluation dataset, this will be a dictionary
                  that contains a set of assessments in the format of
                  [assessment name]: [assessment value].

            * - ``trace``
              - A trace object corresponding to the prediction for the row.
              - Specified as a ``trace`` column in the dataset, or generated during the prediction.

        Example:

            .. code-block:: python

                class NotEmpty(BaseScorer):
                    name = "not_empty"

                    def __call__(self, *, outputs) -> bool:
                        return outputs != ""


                class ExactMatch(BaseScorer):
                    name = "exact_match"

                    def __call__(self, *, outputs, expectations) -> bool:
                        return outputs == expectations["expected_response"]


                class NumToolCalls(BaseScorer):
                    name = "num_tool_calls"

                    def __call__(self, *, trace) -> int:
                        spans = trace.search_spans(name="tool_call")
                        return len(spans)


                # Use the scorer in an evaluation
                mlflow.genai.evaluate(
                    data=data,
                    scorers=[NotEmpty(), ExactMatch(), NumToolCalls()],
                )
        """
        raise NotImplementedError("Implementation of __call__ is required for Scorer class")


@experimental
def scorer(
    func=None,
    *,
    name: Optional[str] = None,
    aggregations: Optional[
        list[Union[Literal["min", "max", "mean", "median", "variance", "p90", "p99"], Callable]]
    ] = None,
):
    """
    A decorator to define a custom scorer that can be used in ``mlflow.genai.evaluate()``.

    The scorer function should take in a **subset** of the following parameters:

    .. list-table::
        :widths: 20 20 20
        :header-rows: 1

        * - Parameter
          - Description
          - Source

        * - ``inputs``
          - A single input to the target model/app.
          - Derived from either dataset or trace.

            * When the dataset contains ``inputs`` column, the value will be passed as is.
            * When traces are provided as evaluation dataset, this will be derived
              from the ``inputs`` field of the trace (i.e. inputs captured as the
              root span of the trace).

        * - ``outputs``
          - A single output from the target model/app.
          - Derived from either dataset, trace, or output of ``predict_fn``.

            * When the dataset contains ``outputs`` column, the value will be passed as is.
            * When ``predict_fn`` is provided, MLflow will make a prediction using the
              ``inputs`` and the ``predict_fn`` and pass the result as the ``outputs``.
            * When traces are provided as evaluation dataset, this will be derived
              from the ``response`` field of the trace (i.e. outputs captured as the
              root span of the trace).

        * - ``expectations``
          - Ground truth or any expectation for each prediction e.g., expected retrieved docs.
          - Derived from either dataset or trace.

            * When the dataset contains ``expectations`` column, the value will be passed as is.
            * When traces are provided as evaluation dataset, this will be a dictionary
              that contains a set of assessments in the format of
              [assessment name]: [assessment value].

        * - ``trace``
          - A trace object corresponding to the prediction for the row.
          - Specified as a ``trace`` column in the dataset, or generated during the prediction.

    The scorer function should return one of the following:

    * A boolean value
    * An integer value
    * A float value
    * A string value
    * A single :class:`~mlflow.entities.Feedback` object
    * A list of :class:`~mlflow.entities.Feedback` objects

    .. note::

        The metric name will be determined by the scorer function's name or a custom name
        specified in the `name` parameter for the scorer.

    Example:

        .. code-block:: python

            import json
            from mlflow.genai.scorers import scorer
            from mlflow.entities import AssessmentSource, Feedback


            # Basic scorers that returns primitive values
            @scorer
            def not_empty(outputs) -> bool:
                return outputs != ""


            @scorer
            def exact_match(outputs, expectations) -> bool:
                return outputs == expectations["expected_response"]


            @scorer
            def num_tool_calls(trace) -> int:
                spans = trace.search_spans(name="tool_call")
                return len(spans)


            # Use `Feedback` object to return additional information about the scorer's
            # result, such as a rationale for the score.
            @scorer
            def harmfulness(outputs) -> Feedback:
                import openai

                prompt = f'''
                    Judge if the following text is harmful or not.

                    Text:
                    {outputs}

                    Return the answer in a JSON object with the following format:
                    {{
                        "harmful": true
                        "reason": "The text contains harmful content"
                    }}

                    Do not output any other characters than the json object.
                '''
                response = openai.OpenAI().chat.completions.create(
                    model="o4-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                payload = json.loads(response.choices[0].message.content)
                return Feedback(
                    value=payload["harmful"],
                    rationale=payload["reason"],
                    source=AssessmentSource(
                        source_type="LLM_JUDGE",
                        source_id="openai:/o4-mini",
                    ),
                )


            # Use the scorer in an evaluation
            mlflow.genai.evaluate(
                data=data,
                scorers=[not_empty, exact_match, num_tool_calls, harmfulness],
            )
    """

    if func is None:
        return functools.partial(scorer, name=name, aggregations=aggregations)

    class CustomScorer(Scorer):
        # Store reference to the original function
        _original_func: Optional[Callable] = PrivateAttr(default=None)

        def __init__(self, **data):
            super().__init__(**data)

        def model_post_init(self, __context: Any) -> None:
            """Set the original function and extract source code."""
            # Set the original function first
            # Use object.__setattr__ to bypass Pydantic's attribute handling for private attributes
            # during model initialization, as direct assignment (self._original_func = func) may be
            # ignored or fail in this context
            object.__setattr__(self, "_original_func", func)
            # Now call the parent's model_post_init
            super().model_post_init(__context)

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

    # Update the __call__ method's signature to match the original function
    # but add 'self' as the first parameter. This is required for MLflow to
    # pass the correct set of parameters to the scorer.
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    new_params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + params
    new_signature = signature.replace(parameters=new_params)
    CustomScorer.__call__.__signature__ = new_signature

    return CustomScorer(
        name=name or func.__name__,
        aggregations=aggregations,
    )
