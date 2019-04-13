# Adding `examples` unit tests to `pytest` test suite 

Two types of test runs for code in `examples` directory are supported:
  * Examples run by `mlflow run`
  * Examples run by another command, such as the `python` interpreter
  
Each of these types of runs are implemented using `@pytest.mark.parametrize` decorator.  Adding a new
example to test involves updating the decorator list as described below. 

For purpose of discussion, `new_example_dir` designates the
directory the example code is found, i.e., it is located in `examples/new_example_dir`.

## Examples that utilize `mlflow run` construct
The `@pytest.mark.mark.parametrize` decorator for `def test_mlflow_run_example(tracking_uri_mock, directory, params):` 
is updated.

If the example is executed by `cd examples/new_example_dir && mflow run . -P parm1=99 -P parm2=3`, then
this `tuple` is added to the decorator list
```
("new_example_dir", ["-P", "parm1=123", "-P", "parm2=99"])
```

as shown below

```
@pytest.mark.parametrize("directory, params", [
    ("sklearn_elasticnet_wine", ["-P", "alpha=0.5"]),
    (os.path.join("sklearn_elasticnet_diabetes", "linux"), []),
    ("new_example_dir", ["-P", "parm1=123", "-P", "parm2=99"]),
])
def test_mlflow_run_example(tracking_uri_mock, directory, params):
```

The `tuple` for an example requiring no parameters is simply:
```
("new_example_dir", []),
```


## Examples that are executed with another command
For an example that is not run by `mlflow run`, the list in 
`@pytest.mark.parametrize` decorator for `test_command_example(tmpdir, directory, command):` is updated.

Examples invoked by `cd examples/new_example_dir && python train.py` require this tuple added
to the decorator's list
```
("new_example_dir", ["python", "train.py"]),
```

as shown below

```
@pytest.mark.parametrize("directory, command", [
    ('sklearn_logistic_regression', ['python', 'train.py']),
    ('h2o', ['python', 'random_forest.py']),
    ('quickstart', ['python', 'mlflow_tracking.py']),
    ("new_example_dir", ["python", "train.py"]),
])
def test_command_example(tmpdir, directory, command):
```

If the example requires arguments to run, i.e., `python train.py arg1 arg2`, then the 
tuple would look like this
```
('new_example_dir', ['python', 'train.py', 'arg1', 'arg2'])
```