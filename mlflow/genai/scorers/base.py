import functools
import inspect
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr

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

    def model_post_init(self, __context: Any) -> None:
        """Extract and store source code after initialization."""
        super().model_post_init(__context)
        
        # Skip source code extraction for builtin scorers
        # Check if the instance is a subclass of BuiltInScorer (without importing it to avoid circular imports)
        if hasattr(self.__class__, '__module__') and 'builtin_scorers' in self.__class__.__module__:
            return
            
        self._extract_source_code()
    
    def _extract_source_code(self):
        """Extract source code for run and __call__ methods."""
        from mlflow.genai.scorers.utils import extract_function_body
        
        # Extract run method source
        try:
            run_body, _ = extract_function_body(self.run)
            self._run_source = run_body
        except Exception:
            # Fallback to inspect.getsource if custom extraction fails
            try:
                self._run_source = inspect.getsource(self.run)
            except Exception:
                self._run_source = None
        
        # Extract __call__ method source
        # Check if this scorer has an _original_func (from @scorer decorator)
        if hasattr(self, '_original_func') and self._original_func:
            try:
                call_body, _ = extract_function_body(self._original_func)
                self._call_source = call_body
            except Exception:
                try:
                    self._call_source = inspect.getsource(self._original_func)
                except Exception:
                    self._call_source = None
        else:
            # Regular extraction for direct Scorer subclasses
            call_method = self.__call__
            
            # Look for the wrapped function in the closure
            if hasattr(call_method, '__closure__') and call_method.__closure__:
                for cell in call_method.__closure__:
                    try:
                        cell_contents = cell.cell_contents
                        if callable(cell_contents) and hasattr(cell_contents, '__name__'):
                            # This might be the wrapped function
                            call_body, _ = extract_function_body(cell_contents)
                            self._call_source = call_body
                            break
                    except Exception:
                        continue
            
            # If we didn't find the wrapped function, try the regular extraction
            if not hasattr(self, '_call_source') or not self._call_source:
                try:
                    call_body, _ = extract_function_body(call_method)
                    self._call_source = call_body
                except Exception:
                    # Fallback to inspect.getsource if custom extraction fails
                    try:
                        self._call_source = inspect.getsource(call_method)
                    except Exception:
                        self._call_source = None
        
        # Store the signature of __call__ for reconstruction
        try:
            self._call_signature = str(inspect.signature(self.__call__))
        except Exception:
            self._call_signature = None

    def model_dump(self, **kwargs) -> dict:
        """Override model_dump to include source code."""
        data = super().model_dump(**kwargs)
        
        # Add source code to the serialized data if available
        if hasattr(self, '_run_source') and self._run_source:
            data["run_source"] = self._run_source
        if hasattr(self, '_call_source') and self._call_source:
            data["__call___source"] = self._call_source
        if hasattr(self, '_call_signature') and self._call_signature:
            data["__call___signature"] = self._call_signature
            
        return data
    
    @classmethod
    def model_validate(cls, obj: Any) -> "Scorer":
        """Override model_validate to reconstruct methods from source code."""
        if isinstance(obj, dict):
            # Extract source code fields before creating instance
            run_source = obj.pop("run_source", None)
            call_source = obj.pop("__call___source", None)
            call_signature = obj.pop("__call___signature", None)
            
            # Check if this is a built-in scorer class
            is_builtin = hasattr(cls, '__module__') and 'builtin_scorers' in cls.__module__
            
            # If we have source code and it's not a builtin scorer, use DynamicScorer
            if call_source and call_signature and not is_builtin:
                # Import here to avoid circular dependency
                from mlflow.genai.scorers.base import DynamicScorer
                
                # Add private attributes to obj for DynamicScorer
                obj['_run_source'] = run_source
                obj['_call_source'] = call_source
                obj['_call_signature'] = call_signature
                
                # Create a DynamicScorer instance
                instance = DynamicScorer(**obj)
                
                # DynamicScorer will handle method reconstruction in its __init__
                return instance
            else:
                # Create regular instance
                instance = super().model_validate(obj)
                
                # Only set source code attributes if not a builtin scorer
                if not is_builtin:
                    try:
                        # Store the source code in private attributes
                        instance._run_source = run_source
                        instance._call_source = call_source
                        instance._call_signature = call_signature
                    except Exception:
                        # Some other immutable scorer - skip source code storage
                        pass
                
                return instance
        return super().model_validate(obj)
    
    def _reconstruct_call_method(self):
        """Reconstruct the __call__ method from stored source code."""
        if not self._call_source or not self._call_signature:
            return
            
        try:
            # Parse the signature to get parameter names
            import re
            sig_match = re.match(r'\((.*?)\)', self._call_signature)
            if not sig_match:
                return
                
            params = [p.strip() for p in sig_match.group(1).split(',') if p.strip()]
            param_str = ', '.join(params)
            
            # Build the function definition
            func_def = f"def __call__(self, {param_str}):\n"
            
            # Indent the source code
            indented_source = '\n'.join(f"    {line}" for line in self._call_source.split('\n'))
            func_def += indented_source
            
            # Create a namespace with necessary imports
            namespace = {
                'Feedback': Feedback,
                'Assessment': Assessment,
                'Union': Union,
                'Optional': Optional,
                'Any': Any,
                'list': list,
                'dict': dict,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'len': len,
                'sum': sum,
                'enumerate': enumerate,
                're': __import__('re'),
            }
            
            # Execute the function definition
            exec(func_def, namespace)
            
            # Get the function from the namespace
            call_func = namespace['__call__']
            
            # Create a bound method and set it on the instance
            # Use object.__setattr__ to bypass pydantic's setattr
            import types
            bound_method = types.MethodType(call_func, self)
            object.__setattr__(self, '__call__', bound_method)
            
        except Exception as e:
            # If reconstruction fails, keep the original NotImplementedError behavior
            import warnings
            warnings.warn(f"Failed to reconstruct __call__ method: {e}")

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
class DynamicScorer(Scorer):
    """A scorer that can reconstruct its __call__ method from source code."""
    
    def __init__(self, **data):
        # Extract source code fields if present and save them
        run_source = data.pop('_run_source', None)
        call_source = data.pop('_call_source', None)
        call_signature = data.pop('_call_signature', None)
        
        # Now call parent init
        super().__init__(**data)
        
        # Re-set the source code attributes after parent init
        # Use object.__setattr__ to bypass any pydantic restrictions
        if run_source is not None:
            object.__setattr__(self, '_run_source', run_source)
        if call_source is not None:
            object.__setattr__(self, '_call_source', call_source)
        if call_signature is not None:
            object.__setattr__(self, '_call_signature', call_signature)
        
        # Set up the dynamic call method if we have source code
        if self._call_source and self._call_signature:
            self._setup_dynamic_call()
    
    def model_post_init(self, __context: Any) -> None:
        """Override to prevent source code extraction for dynamic scorers."""
        # Skip parent's model_post_init which would extract source code
        # We already have the source code from deserialization
        pass
    
    def _setup_dynamic_call(self):
        """Set up the dynamic __call__ method from source code."""
        if not self._call_source or not self._call_signature:
            return
            
        try:
            # Parse the signature to get parameter names
            import re
            sig_match = re.match(r'\((.*?)\)', self._call_signature)
            if not sig_match:
                return
                
            params_str = sig_match.group(1).strip()
            
            # Build a function that matches the expected signature
            # Note: params_str already contains the parameter definitions
            # Make sure to add * to enforce keyword-only parameters if not already present
            if params_str and not params_str.startswith('*'):
                func_def = f"def __call__(self, *, {params_str}):\n"
            else:
                func_def = f"def __call__(self, {params_str}):\n"
            
            # Indent the source code
            indented_source = '\n'.join(f"    {line}" for line in self._call_source.split('\n'))
            func_def += indented_source
            
            # Create a namespace with necessary imports
            namespace = {
                'Feedback': Feedback,
                'Assessment': Assessment,
                'Union': Union,
                'Optional': Optional,
                'Any': Any,
                'list': list,
                'dict': dict,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'len': len,
                'sum': sum,
                'enumerate': enumerate,
                're': __import__('re'),
            }
            
            # Execute the function definition
            exec(func_def, namespace)
            
            # Get the function and bind it
            import types
            call_func = namespace['__call__']
            bound_method = types.MethodType(call_func, self)
            
            # Use object.__setattr__ to bypass pydantic
            object.__setattr__(self, '_dynamic_call_func', bound_method)
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to set up dynamic call: {e}")

    def __call__(
        self, 
        *, 
        inputs: Any = None, 
        outputs: Any = None, 
        expectations: Optional[dict[str, Any]] = None, 
        trace: Optional[Trace] = None
    ) -> Union[int, float, bool, str, Feedback, list[Feedback]]:
        """Call the dynamically loaded function with appropriate parameters."""
        if hasattr(self, '_dynamic_call_func') and self._dynamic_call_func:
            # Build kwargs with only the parameters the dynamic function expects
            try:
                sig = inspect.signature(self._dynamic_call_func)
                kwargs = {}
                all_params = {
                    'inputs': inputs,
                    'outputs': outputs,
                    'expectations': expectations,
                    'trace': trace
                }
                for param_name in sig.parameters:
                    if param_name != 'self' and param_name in all_params:
                        kwargs[param_name] = all_params[param_name]
                        
                # Call our dynamically loaded function
                return self._dynamic_call_func(**kwargs)
            except Exception as e:
                raise
        else:
            # Fall back to parent's implementation
            return super().__call__(
                inputs=inputs, 
                outputs=outputs, 
                expectations=expectations, 
                trace=trace
            )


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
            self._original_func = func
        
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