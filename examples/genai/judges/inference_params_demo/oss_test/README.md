# OSS Test: inference_params with LiteLLM (OpenAI)

This folder contains the OSS (non-Databricks) test for the `inference_params` feature using the LiteLLM adapter with OpenAI.

## Test Script

`test_oss_litellm.py` - Tests inference_params with OpenAI via LiteLLM

## Running the Test

```bash
export OPENAI_API_KEY=<your-key>
python3 test_oss_litellm.py
```

## Test Output

See `oss_test_output.txt` for the actual test run output.

## Test Results Summary

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | Deterministic (temp=0.0) | ✓ PASS - All 3 rationales identical |
| Test 2 | Varied (temp=1.0) | ✓ PASS - Rationales varied across runs |
| Test 3 | Multiple params | ✓ PASS - All params applied |
| Test 4 | Default (no params) | ✓ PASS - inference_params=None |

## Key Results

### Test 1: Deterministic (temperature=0.0)
```
Run 1: accurate - The Eiffel Tower is 330 meters tall is factually accurate...
Run 2: accurate - The Eiffel Tower is 330 meters tall is factually accurate...
Run 3: accurate - The Eiffel Tower is 330 meters tall is factually accurate...

[Result] All rationales identical: True
```

### Test 2: Varied (temperature=1.0)
```
Run 1: accurate - The Eiffel Tower's height is commonly cited as approximately 330 meters...
Run 2: accurate - The provided output states that the Eiffel Tower is 330 meters tall...
Run 3: accurate - The statement that the Eiffel Tower is 330 meters tall is factually accurate...

[Result] All rationales identical: False
```

## Environment

- Model: `openai:/gpt-4o-mini`
- Adapter: LiteLLM
- MLflow: Local development build (PR #19152)
