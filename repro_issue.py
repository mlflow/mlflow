
import pandas as pd

from typing import List
from mlflow.types.type_hints import _convert_data_to_type_hint

def test_convert_data_to_type_hint_warning():
    # Create a DataFrame with multiple columns
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    
    # Type hint expectation: list[int] (single column implied)
    type_hint = List[int]
    
    # This should verify if the warning is triggered
    print("Calling _convert_data_to_type_hint with multi-column DF and list[int]")
    result = _convert_data_to_type_hint(df, type_hint)
    print(f"Result: {result}")

if __name__ == "__main__":
    test_convert_data_to_type_hint_warning()
