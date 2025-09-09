# MLflow Test Duration Files

This directory contains test duration data used by MLflow's intelligent test parallelization system powered by [pytest-split](https://github.com/jerry-git/pytest-split).

## How It Works

MLflow uses test duration data to intelligently distribute tests across parallel CI jobs, significantly reducing overall CI runtime by ensuring each parallel job takes approximately the same amount of time.

### The Feedback Loop

1. **Read Path (CI Execution)**:

   - CI jobs read duration files (`{job}.test_duration`) to make intelligent splitting decisions
   - pytest-split uses this historical timing data to distribute tests evenly across parallel groups
   - Each job runs its assigned subset of tests with `--splits=N --group=X`

2. **Write Path (Duration Collection)**:

   - During test execution, pytest-split collects new timing data with `--store-durations`
   - Each parallel job uploads its timing data as CI artifacts (`test-durations-{job}-group-{N}`)
   - The [`download_test_durations.py`](../../../dev/download_test_durations.py) script downloads these artifacts
   - New timing data is merged into the consolidated duration files in this directory

3. **Continuous Improvement**:
   - Each CI run improves future test distribution with updated timing data
   - This creates a feedback loop where test parallelization becomes more efficient over time

## File Structure

Each file corresponds to a CI job and contains JSON mapping of test names to execution times:

```
{job}.test_duration
├── python.test_duration          # Core Python tests
├── pyfunc.test_duration          # PyFunc model tests
├── models.test_duration          # Model management tests
├── evaluate.test_duration        # Model evaluation tests
├── genai.test_duration          # GenAI/LLM tests
├── flavors.test_duration        # ML framework integration tests
├── sagemaker.test_duration      # AWS SageMaker tests
├── windows.test_duration        # Windows compatibility tests
├── python-skinny-tests.test_duration  # Minimal dependency tests
└── pyfunc-pydanticv1.test_duration    # Pydantic v1 compatibility tests
```

### Example File Format

```json
{
  "tests/models/test_model.py::test_log_model": 2.45,
  "tests/models/test_model.py::test_load_model": 1.23,
  "tests/tracking/test_tracking.py::test_create_experiment": 0.87
}
```

## Integration Points

### CI Workflow Integration

The duration files are used by the [`manage-test-durations`](../actions/manage-test-durations/action.yml) composite action:

```yaml
- uses: ./.github/actions/manage-test-durations
  with:
    job_name: "python"
    test_command: "uv run pytest tests/"
```

This action handles:

1. **Reading**: Copying the consolidated duration file for pytest-split
2. **Execution**: Running tests with duration collection enabled
3. **Writing**: Uploading new timing data as CI artifacts

### Manual Updates

To manually update duration files from CI runs:

```bash
# Download from latest master run
python dev/download_test_durations.py

# Download from specific run ID
python dev/download_test_durations.py --run-id 1234567890

# Download from feature branch
python dev/download_test_durations.py --branch feature-branch
```

## Benefits

- **Faster CI**: Tests distributed evenly across parallel jobs
- **Automatic Optimization**: Duration data improves over time without manual intervention
- **Reduced Flakiness**: More predictable job completion times
- **Scalable**: Easy to add new jobs or adjust parallelization levels

## Maintenance

Duration files are automatically updated through CI feedback loops. Manual updates are rarely needed unless:

- Adding new CI jobs that need duration bootstrapping
- Major test suite restructuring
- Debugging CI performance issues

For questions or issues, see:

- [pytest-split documentation](https://github.com/jerry-git/pytest-split)
- [`dev/download_test_durations.py`](../../../dev/download_test_durations.py) for CI artifact collection
- [`.github/actions/manage-test-durations/`](../actions/manage-test-durations/) for CI integration
