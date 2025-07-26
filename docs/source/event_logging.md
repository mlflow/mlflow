# Event & Stage Logging API

MLflow now supports logging custom events and pipeline stages during a run, and visualizing them as timeline annotations in the System Metrics UI.

## API Usage

### Manual Event/Stage Logging
```python
mlflow.log_event(name="event_name")
mlflow.log_event(name="stage_name", start_time=..., time=...)
```

### Automatic Stage Logging
```python
with mlflow.log_stage("data_loading"):
    ... # code
```

## UI
- Events are shown as vertical lines with hover text.
- Stages are shown as shaded regions labeled with the stage name.
- Legends and filters allow toggling visibility of events/stages.

## Storage
Events and stages are stored as a JSON artifact (`events.json`) in the run.
