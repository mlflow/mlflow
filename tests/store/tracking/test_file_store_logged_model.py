import time
import uuid
from dataclasses import dataclass
from unittest import mock

import pytest

import mlflow
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT, FileStore


@pytest.fixture
def store(tmp_path):
    return FileStore(str(tmp_path.joinpath("mlruns")))


@dataclass
class DummyDataset:
    name: str
    digest: str


def assert_logged_model_attributes(
    logged_model,
    experiment_id,
    name=None,
    source_run_id=None,
    tags=None,
    params=None,
    model_type=None,
    status=str(LoggedModelStatus.PENDING),
):
    assert logged_model.experiment_id == experiment_id
    if name is None:
        assert logged_model.name is not None
    else:
        assert logged_model.name == name
    if source_run_id is None:
        assert logged_model.source_run_id is None
    else:
        assert logged_model.source_run_id == source_run_id
    # NB: In many cases, there are default tags populated from the context in which a model
    # is created, such as the notebook ID or git hash. We don't want to assert that these
    # tags are present in the model's tags field (because it's complicated and already tested
    # elsewhere), so we only check that the tags we specify are present
    assert (tags or {}).items() <= logged_model.tags.items()
    assert logged_model.params == (params or {})
    assert logged_model.model_type == model_type
    assert logged_model.status == status


def assert_models_match(models1, models2):
    assert len(models1) == len(models2)
    m1 = sorted([m.to_dictionary() for m in models1], key=lambda m: m["model_id"])
    m2 = sorted([m.to_dictionary() for m in models2], key=lambda m: m["model_id"])
    assert m1 == m2


def test_initialize_logged_model_when_set_experiment():
    exp = mlflow.set_experiment("test")
    logged_model = mlflow.initialize_logged_model(exp.experiment_id)
    assert_logged_model_attributes(
        logged_model,
        exp.experiment_id,
    )


def test_create_logged_model(store):
    logged_model = store.create_logged_model()
    assert_logged_model_attributes(
        logged_model,
        FileStore.DEFAULT_EXPERIMENT_ID,
    )

    exp_id = store.create_experiment("test")
    logged_model = store.create_logged_model(exp_id)
    assert_logged_model_attributes(
        logged_model,
        exp_id,
    )

    run_id = store.create_run(
        exp_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="test_run",
    ).info.run_id
    logged_model = store.create_logged_model(exp_id, source_run_id=run_id)
    assert_logged_model_attributes(
        logged_model,
        exp_id,
        source_run_id=run_id,
    )

    logged_model = store.create_logged_model(
        exp_id,
        name="test_model",
        source_run_id=run_id,
    )
    assert_logged_model_attributes(
        logged_model,
        exp_id,
        name="test_model",
        source_run_id=run_id,
    )

    logged_model = store.create_logged_model(
        exp_id,
        name="test_model",
        source_run_id=run_id,
        tags=[LoggedModelTag("tag_key", "tag_value")],
        params=[LoggedModelParameter("param_key", "param_value")],
    )
    assert_logged_model_attributes(
        logged_model,
        exp_id,
        name="test_model",
        source_run_id=run_id,
        tags={"tag_key": "tag_value"},
        params={"param_key": "param_value"},
    )


def test_log_logged_model_params(store):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    assert not model.params
    store.log_logged_model_params(
        model_id=model.model_id, params=[LoggedModelParameter("param1", "apple")]
    )
    loaded_model = store.get_logged_model(model_id=model.model_id)
    assert loaded_model.params == {"param1": "apple"}


def test_create_logged_model_errors(store):
    with pytest.raises(MlflowException, match=r"Could not find experiment with ID 123"):
        store.create_logged_model("123")
    exp_id = store.create_experiment("test")
    store.delete_experiment(exp_id)
    with pytest.raises(
        MlflowException,
        match=rf"Could not create model under non-active experiment with ID {exp_id}",
    ):
        store.create_logged_model(exp_id)

    with pytest.raises(MlflowException, match=r"A key name must be provided."):
        store.create_logged_model(params=[LoggedModelParameter(None, "b")])

    with pytest.raises(MlflowException, match=r"exceeds the maximum length"):
        store.create_logged_model(params=[LoggedModelParameter("a" * 256, "b")])


@pytest.mark.parametrize(
    "name",
    [
        "",
        "my/model",
        "my.model",
        "my:model",
        "my%model",
        "my'model",
        'my"model',
    ],
)
def test_create_logged_model_invalid_name(store: FileStore, name: str):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    with pytest.raises(MlflowException, match="Invalid model name"):
        store.create_logged_model(exp_id, name=name)


def test_set_logged_model_tags(store):
    exp_id = store.create_experiment("test")
    run_id = store.create_run(exp_id, "user", 0, [], "test_run").info.run_id
    logged_model = store.create_logged_model(exp_id, "test_model", run_id)
    assert logged_model.tags == {}
    store.set_logged_model_tags(logged_model.model_id, [LoggedModelTag("tag_key", "tag_value")])
    logged_model = store.get_logged_model(logged_model.model_id)
    assert logged_model.tags == {"tag_key": "tag_value"}
    store.set_logged_model_tags(logged_model.model_id, [LoggedModelTag("tag_key", "new_tag_value")])
    logged_model = store.get_logged_model(logged_model.model_id)
    assert logged_model.tags == {"tag_key": "new_tag_value"}
    store.set_logged_model_tags(
        logged_model.model_id, [LoggedModelTag("a", None), LoggedModelTag("b", 123)]
    )
    logged_model = store.get_logged_model(logged_model.model_id)
    assert logged_model.tags == {"tag_key": "new_tag_value", "a": "", "b": "123"}


def test_set_logged_model_tags_errors(store):
    logged_model = store.create_logged_model()
    with pytest.raises(MlflowException, match=r"Missing value for required parameter"):
        store.set_logged_model_tags(logged_model.model_id, [LoggedModelTag(None, None)])
    with pytest.raises(MlflowException, match=r"Names may only contain alphanumerics"):
        store.set_logged_model_tags(logged_model.model_id, [LoggedModelTag("a!b", "c")])


def test_delete_logged_model_tag(store: FileStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    model = store.create_logged_model(experiment_id=exp_id)
    store.set_logged_model_tags(model.model_id, [LoggedModelTag("tag1", "apple")])
    store.delete_logged_model_tag(model.model_id, "tag1")
    assert store.get_logged_model(model.model_id).tags == {}

    with pytest.raises(MlflowException, match="not found"):
        store.delete_logged_model_tag("does-not-exist", "tag1")

    with pytest.raises(MlflowException, match="No tag with key"):
        store.delete_logged_model_tag(model.model_id, "tag1")


def test_get_logged_model(store):
    experiment_id = store.create_experiment("test")
    run_id = store.create_run(
        experiment_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="test_run",
    ).info.run_id
    logged_model = store.create_logged_model(
        experiment_id=experiment_id,
        name="test_model",
        source_run_id=run_id,
        tags=[LoggedModelTag("tag_key", "tag_value")],
        params=[LoggedModelParameter("param_key", "param_value")],
        model_type="dev",
    )
    fetched_model = store.get_logged_model(logged_model.model_id)
    assert logged_model.model_uri == fetched_model.model_uri
    assert logged_model.to_dictionary() == fetched_model.to_dictionary()


def test_delete_logged_model(store: FileStore):
    exp_id = store.create_experiment("test")
    run = store.create_run(exp_id, "user", 0, [], "test_run")
    logged_model = store.create_logged_model(experiment_id=exp_id, source_run_id=run.info.run_id)
    metric = Metric(
        key="metric",
        value=0,
        timestamp=0,
        step=0,
        model_id=logged_model.model_id,
        run_id=run.info.run_id,
    )
    store.log_metric(run.info.run_id, metric)
    assert store.get_logged_model(logged_model.model_id) is not None
    store.delete_logged_model(logged_model.model_id)
    with pytest.raises(MlflowException, match=r"not found"):
        store.get_logged_model(logged_model.model_id)

    models = store.search_logged_models(experiment_ids=["0"])
    assert len(models) == 0


def test_get_logged_model_errors(store):
    with pytest.raises(MlflowException, match=r"Model '1234' not found"):
        store.get_logged_model("1234")

    with (
        mock.patch(
            "mlflow.store.tracking.file_store.FileStore._find_model_root",
            return_value=("0", "abc"),
        ),
        mock.patch(
            "mlflow.store.tracking.file_store.FileStore._get_model_info_from_dir",
            return_value={"experiment_id": "1"},
        ),
    ):
        with pytest.raises(MlflowException, match=r"Model '1234' metadata is in invalid state"):
            store.get_logged_model("1234")


def test_finalize_logged_model(store):
    logged_model = store.create_logged_model()
    assert logged_model.status == str(LoggedModelStatus.PENDING)
    time.sleep(0.001)  # sleep to ensure last_updated_timestamp is updated
    updated_model = store.finalize_logged_model(logged_model.model_id, LoggedModelStatus.READY)
    assert updated_model.status == str(LoggedModelStatus.READY)
    assert logged_model.experiment_id == updated_model.experiment_id
    assert logged_model.model_id == updated_model.model_id
    assert logged_model.name == updated_model.name
    assert logged_model.artifact_location == updated_model.artifact_location
    assert logged_model.creation_timestamp == updated_model.creation_timestamp
    assert logged_model.last_updated_timestamp < updated_model.last_updated_timestamp
    assert logged_model.model_type == updated_model.model_type
    assert logged_model.source_run_id == updated_model.source_run_id
    assert logged_model.tags == updated_model.tags
    assert logged_model.params == updated_model.params
    assert logged_model.metrics == updated_model.metrics

    updated_model = store.finalize_logged_model(logged_model.model_id, LoggedModelStatus.FAILED)
    assert updated_model.status == str(LoggedModelStatus.FAILED)


def test_finalize_logged_model_errors(store):
    with pytest.raises(MlflowException, match=r"Model '1234' not found"):
        store.finalize_logged_model("1234", LoggedModelStatus.READY)


def test_search_logged_models_experiment_ids(store):
    exp_ids = []
    for i in range(5):
        exp_id = store.create_experiment(f"test_{i}")
        store.create_logged_model(exp_id)
        exp_ids.append(exp_id)
    assert len(store.search_logged_models(experiment_ids=exp_ids)) == 5
    for exp_id in exp_ids:
        assert len(store.search_logged_models(experiment_ids=[exp_id])) == 1
    assert len(store.search_logged_models(experiment_ids=[])) == 0
    assert len(store.search_logged_models(experiment_ids=exp_ids + ["1234"])) == 5


def test_search_logged_models_filter_string(store):
    exp_id = store.create_experiment("test")
    run_ids = []
    logged_models = []
    for i in range(5):
        run_ids.append(store.create_run(exp_id, "user", 0, [], f"test_run_{i}").info.run_id)
        logged_models.append(
            store.create_logged_model(exp_id, source_run_id=run_ids[-1], model_type="test")
        )
        # make sure the creation_timestamp is different
        time.sleep(0.001)
    logged_models = sorted(logged_models, key=lambda x: (-x.creation_timestamp, x.model_id))
    run_ids = run_ids[::-1]

    # model_id
    for model in logged_models:
        models = store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"model_id='{model.model_id}'"
        )
        assert len(models) == 1
        assert models[0].to_dictionary() == model.to_dictionary()

    model = logged_models[0]
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"model_id!='{model.model_id}'"
    )
    assert_models_match(models, logged_models[1:])

    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"model_id LIKE '{model.model_id}'"
    )
    assert_models_match(models, [model])

    models = store.search_logged_models(
        experiment_ids=[exp_id],
        filter_string=f"model_id ILIKE '{model.model_id.upper()}'",
    )
    assert_models_match(models, [model])

    # name
    for model in logged_models:
        models = store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"name='{model.name}'"
        )
        assert len(models) == 1
        assert models[0].to_dictionary() == model.to_dictionary()

    model = logged_models[0]
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name!='{model.name}'"
    )
    assert_models_match(models, logged_models[1:])

    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name LIKE '{model.name}'"
    )
    assert_models_match(models, [model])

    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name ILIKE '{model.name.upper()}'"
    )
    assert_models_match(models, [model])

    # model_type
    models = store.search_logged_models(experiment_ids=[exp_id], filter_string="model_type='test'")
    assert_models_match(models, logged_models)
    models = store.search_logged_models(experiment_ids=[exp_id], filter_string="model_type!='test'")
    assert len(models) == 0
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string="model_type LIKE 'te%'"
    )
    assert_models_match(models, logged_models)
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string="model_type ILIKE 'TE%'"
    )
    assert_models_match(models, logged_models)

    # status
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"status='{LoggedModelStatus.PENDING}'"
    )
    assert_models_match(models, logged_models)
    updated_model = store.finalize_logged_model(logged_models[0].model_id, LoggedModelStatus.READY)
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"status!='{LoggedModelStatus.PENDING}'"
    )
    logged_models = [updated_model, *logged_models[1:]]
    assert_models_match(models, [logged_models[0]])
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"status LIKE '{LoggedModelStatus.READY}'"
    )
    assert_models_match(models, [logged_models[0]])
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string="status ILIKE 'ready'"
    )
    assert_models_match(models, [logged_models[0]])

    # source_run_id
    for i, run_id in enumerate(run_ids):
        models = store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"source_run_id='{run_id}'"
        )
        assert_models_match(models, [logged_models[i]])
    run_id = run_ids[0]
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"source_run_id!='{run_id}'"
    )
    assert_models_match(models, logged_models[1:])
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"source_run_id LIKE '{run_id}'"
    )
    assert_models_match(models, [logged_models[0]])
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"source_run_id ILIKE '{run_id.upper()}'"
    )
    assert_models_match(models, [logged_models[0]])

    # creation_timestamp
    mid_time = logged_models[2].creation_timestamp
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"creation_timestamp={mid_time}"
        ),
        [logged_models[2]],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"creation_timestamp<{mid_time}"
        ),
        logged_models[3:],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"creation_timestamp>{mid_time}"
        ),
        logged_models[:2],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"creation_timestamp<={mid_time}"
        ),
        logged_models[2:],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"creation_timestamp>={mid_time}"
        ),
        logged_models[:3],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"creation_timestamp!={mid_time}"
        ),
        logged_models[:2] + logged_models[3:],
    )

    # last_updated_timestamp
    max_time = store.get_logged_model(logged_models[0].model_id).last_updated_timestamp
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"last_updated_timestamp={max_time}"
        ),
        [logged_models[0]],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"last_updated_timestamp<{max_time}"
        ),
        logged_models[1:],
    )
    assert (
        len(
            store.search_logged_models(
                experiment_ids=[exp_id], filter_string=f"last_updated_timestamp>{max_time}"
            )
        )
        == 0
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"last_updated_timestamp<={max_time}"
        ),
        logged_models,
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"last_updated_timestamp>={max_time}"
        ),
        [logged_models[0]],
    )
    assert_models_match(
        store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"last_updated_timestamp!={max_time}"
        ),
        logged_models[1:],
    )

    # tags
    store.set_logged_model_tags(logged_models[0].model_id, [LoggedModelTag("a", "b")])
    updated_model = store.get_logged_model(logged_models[0].model_id)
    logged_models = [updated_model, *logged_models[1:]]
    assert_models_match(
        store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a`='b'"),
        [logged_models[0]],
    )
    assert (
        len(store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a`!='b'")) == 0
    )
    assert_models_match(
        store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a` LIKE 'b'"),
        [logged_models[0]],
    )
    assert_models_match(
        store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a` ILIKE 'B'"),
        [logged_models[0]],
    )

    # and
    run_id = store.create_run(exp_id, "user", 0, [], "test_run_2").info.run_id
    logged_models = []
    for name in ["test_model1", "test_model2"]:
        logged_models.append(store.create_logged_model(exp_id, name, run_id))
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name='test_model1' AND source_run_id='{run_id}'"
    )
    assert_models_match(models, [logged_models[0]])

    model = logged_models[0]
    for val in (
        # A single item without a comma
        f"('{model.name}')",
        # A single item with a comma
        f"('{model.name}',)",
        # Multiple items
        f"('{model.name}', 'foo')",
    ):
        # IN
        models = store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string=f"name IN {val}",
        )
        assert [m.name for m in models] == [model.name]
        assert models.token is None
        # NOT IN
        models = store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string=f"name NOT IN {val}",
        )
        assert model.name not in [m.name for m in models]


def test_search_logged_models_datasets_filter(store):
    exp_id = store.create_experiment("test")
    run_id = store.create_run(exp_id, "user", 0, [], "test_run").info.run_id
    model1 = store.create_logged_model(exp_id, source_run_id=run_id)
    model2 = store.create_logged_model(exp_id, source_run_id=run_id)
    model3 = store.create_logged_model(exp_id, source_run_id=run_id)
    store.log_batch(
        run_id,
        metrics=[
            Metric(
                key="metric1",
                value=0.1,
                timestamp=0,
                step=0,
                model_id=model1.model_id,
                dataset_name="dataset1",
                dataset_digest="digest1",
            ),
            Metric(
                key="metric1",
                value=0.2,
                timestamp=0,
                step=0,
                model_id=model2.model_id,
                dataset_name="dataset1",
                dataset_digest="digest2",
            ),
            Metric(key="metric2", value=0.1, timestamp=0, step=0, model_id=model3.model_id),
        ],
        params=[],
        tags=[],
    )
    model1 = store.get_logged_model(model1.model_id)
    model2 = store.get_logged_model(model2.model_id)
    model3 = store.get_logged_model(model3.model_id)

    # Restrict results to models with metrics on dataset1
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="metrics.metric1 >= 0.1",
        datasets=[{"dataset_name": "dataset1"}],
    )
    assert_models_match(models, [model1, model2])
    # Restrict results to models with metrics on dataset1 and digest1
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="metrics.metric1 >= 0.1",
        datasets=[{"dataset_name": "dataset1", "dataset_digest": "digest1"}],
    )
    assert_models_match(models, [model1])
    # No filter string, match models with any metrics on the dataset
    models = store.search_logged_models(
        experiment_ids=[exp_id], datasets=[{"dataset_name": "dataset1"}]
    )
    assert_models_match(models, [model1, model2])


def test_search_logged_models_order_by(store):
    exp_id = store.create_experiment("test")
    logged_models = []
    for i in range(5):
        run_id = store.create_run(exp_id, "user", 0, [], f"test_run_{i}").info.run_id
        logged_models.append(
            store.create_logged_model(exp_id, source_run_id=run_id, model_type=f"test_{i}")
        )
        # make sure the creation_timestamp is different
        time.sleep(0.001)

    # default: order by creation_timestamp DESC, model_id ASC
    models = store.search_logged_models(experiment_ids=[exp_id])
    assert_models_match(
        models, sorted(logged_models, key=lambda x: (-x.creation_timestamp, x.model_id))
    )

    # model_id
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "model_id"}]
    )
    assert_models_match(models, sorted(logged_models, key=lambda x: x.model_id))
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "model_id", "ascending": False}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: x.model_id, reverse=True),
    )

    # name
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "name", "ascending": True}]
    )
    assert_models_match(
        models, sorted(logged_models, key=lambda x: (x.name, -x.creation_timestamp, x.model_id))
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {"field_name": "name", "ascending": False},
            {"field_name": "model_id", "ascending": False},
        ],
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: x.name, reverse=True),
    )

    # model_type
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "model_type", "ascending": True}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: x.model_type),
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "model_type", "ascending": False}]
    )
    assert_models_match(
        models,
        sorted(
            logged_models,
            key=lambda x: x.model_type,
            reverse=True,
        ),
    )

    # status
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "status", "ascending": True}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (x.status, -x.creation_timestamp, x.model_id)),
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "status", "ascending": False}]
    )
    assert_models_match(
        models,
        sorted(
            # all status the same
            logged_models,
            key=lambda x: (x.status, -x.creation_timestamp, x.model_id),
        ),
    )

    # source_run_id
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "source_run_id", "ascending": True}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (x.source_run_id, -x.creation_timestamp, x.model_id)),
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {"field_name": "source_run_id", "ascending": False},
            {"field_name": "model_id", "ascending": False},
        ],
    )
    assert_models_match(
        models,
        sorted(
            logged_models,
            key=lambda x: x.source_run_id,
            reverse=True,
        ),
    )

    # creation_timestamp
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "creation_timestamp", "ascending": True}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (x.creation_timestamp, x.model_id)),
    )
    # Alias for creation_timestamp
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "creation_time", "ascending": True}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (x.creation_timestamp, x.model_id)),
    )

    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "creation_timestamp", "ascending": False}]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (-x.creation_timestamp, x.model_id)),
    )

    # last_updated_timestamp
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "last_updated_timestamp"}]
    )
    assert_models_match(
        models,
        sorted(
            logged_models,
            key=lambda x: (x.last_updated_timestamp, -x.creation_timestamp, x.model_id),
        ),
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "last_updated_timestamp", "ascending": False}],
    )
    assert_models_match(
        models,
        sorted(
            logged_models,
            key=lambda x: (-x.last_updated_timestamp, -x.creation_timestamp, x.model_id),
        ),
    )


def test_search_logged_models_order_by_metrics(store):
    exp_id = store.create_experiment("test")
    run_id = store.create_run(exp_id, "user", 0, [], "test_run").info.run_id
    model1 = store.create_logged_model(exp_id, source_run_id=run_id)
    # model1:
    # metric1+dataset1+digest1+timestamp0: 0.1
    # metric1+dataset1+digest2+timestamp1: 1.0
    # metric2+dataset2+digest2: timestamp 0 - 0.2; timestamp 1 - 1.0
    store.log_batch(
        run_id,
        metrics=[
            Metric(
                key="metric1",
                value=0.1,
                timestamp=0,
                step=0,
                model_id=model1.model_id,
                dataset_name="dataset1",
                dataset_digest="digest1",
                run_id=run_id,
            ),
            Metric(
                key="metric1",
                value=1.0,
                timestamp=1,
                step=0,
                model_id=model1.model_id,
                dataset_name="dataset1",
                dataset_digest="digest2",
                run_id=run_id,
            ),
            Metric(
                key="metric2",
                value=0.2,
                timestamp=0,
                step=0,
                model_id=model1.model_id,
                dataset_name="dataset2",
                dataset_digest="digest2",
                run_id=run_id,
            ),
            Metric(
                key="metric2",
                value=1.0,
                timestamp=1,
                step=1,
                model_id=model1.model_id,
                dataset_name="dataset2",
                dataset_digest="digest2",
                run_id=run_id,
            ),
        ],
        params=[],
        tags=[],
    )
    time.sleep(0.01)
    model2 = store.create_logged_model(exp_id, source_run_id=run_id)
    # model2:
    # metric1+dataset1+digest1+timestamp0: 0.2
    # metric1+dataset1+digest2+timestamp1: 0.1
    # metric2+dataset2+digest2+timestamp0: 0.5
    # metric2+dataset3+digest3+timestamp1: 0.1
    # metric3: 1.0
    store.log_batch(
        run_id,
        metrics=[
            Metric(
                key="metric1",
                value=0.2,
                timestamp=0,
                step=0,
                model_id=model2.model_id,
                dataset_name="dataset1",
                dataset_digest="digest1",
                run_id=run_id,
            ),
            Metric(
                key="metric1",
                value=0.1,
                timestamp=1,
                step=0,
                model_id=model2.model_id,
                dataset_name="dataset1",
                dataset_digest="digest2",
                run_id=run_id,
            ),
            Metric(
                key="metric2",
                value=0.5,
                timestamp=0,
                step=0,
                model_id=model2.model_id,
                dataset_name="dataset2",
                dataset_digest="digest2",
                run_id=run_id,
            ),
            Metric(
                key="metric2",
                value=0.1,
                timestamp=1,
                step=0,
                model_id=model2.model_id,
                dataset_name="dataset3",
                dataset_digest="digest3",
                run_id=run_id,
            ),
            Metric(
                key="metric3",
                value=1.0,
                timestamp=0,
                step=1,
                model_id=model2.model_id,
                dataset_name=None,
                dataset_digest=None,
                run_id=run_id,
            ),
        ],
        params=[],
        tags=[],
    )
    model1 = store.get_logged_model(model1.model_id)
    model2 = store.get_logged_model(model2.model_id)

    # order by metric key only
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "metrics.metric1"}]
    )
    assert_models_match(models, [model2, model1])
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "metrics.metric1", "ascending": False}]
    )
    assert_models_match(models, [model1, model2])
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "metrics.metric1", "dataset_name": ""}]
    )
    assert_models_match(models, [model2, model1])
    # only model2 has metric3
    # no matter it's ASC or DESC, model2 should be first
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "metrics.metric3"}]
    )
    assert_models_match(models, [model2, model1])
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=[{"field_name": "metrics.metric3", "ascending": False}]
    )
    assert_models_match(models, [model2, model1])
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {"field_name": "metrics.metric1", "ascending": False},
            {"field_name": "metrics.metric3"},
        ],
    )
    assert_models_match(models, [model1, model2])

    # order by metric key + dataset_name
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {"field_name": "metrics.metric1", "dataset_name": "dataset1", "ascending": False}
        ],
    )
    assert_models_match(models, [model1, model2])
    # neither of them have metric1 + dataset2, so the order is by creation_timestamp DESC
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {"field_name": "metrics.metric1", "dataset_name": "dataset2", "ascending": False}
        ],
    )
    assert_models_match(models, [model2, model1])
    # only model2 has metric2+dataset3, so it always comes first
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "metrics.metric2", "dataset_name": "dataset3", "ascending": True}],
    )
    assert_models_match(models, [model2, model1])
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {"field_name": "metrics.metric2", "dataset_name": "dataset3", "ascending": False}
        ],
    )
    assert_models_match(models, [model2, model1])

    # order by metric key + dataset_name + dataset_digest
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.metric1",
                "dataset_name": "dataset1",
                "dataset_digest": "digest1",
            }
        ],
    )
    assert_models_match(models, [model1, model2])
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.metric1",
                "dataset_name": "dataset1",
                "dataset_digest": "digest2",
            }
        ],
    )
    assert_models_match(models, [model2, model1])
    # max value of different steps is taken, so metric2's value of model1 > model2
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.metric2",
                "dataset_name": "dataset2",
                "dataset_digest": "digest2",
                "ascending": False,
            }
        ],
    )
    assert_models_match(models, [model1, model2])
    # neither of them have metric4, so it's sorted based on second condition
    models = store.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[
            {
                "field_name": "metrics.metric4",
            },
            {
                "field_name": "metrics.metric2",
                "dataset_name": "dataset2",
                "dataset_digest": "digest2",
                "ascending": False,
            },
        ],
    )
    assert_models_match(models, [model1, model2])


def test_search_logged_models_pagination(store):
    exp_id = store.create_experiment("test")
    run_id = store.create_run(exp_id, "user", 0, [], "test").info.run_id
    logged_models = []
    for _ in range(SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT + 20):
        logged_models.append(store.create_logged_model(exp_id, source_run_id=run_id))
    logged_models = sorted(logged_models, key=lambda x: (-x.creation_timestamp, x.model_id))
    models = store.search_logged_models(experiment_ids=[exp_id])
    assert_models_match(
        models,
        logged_models[:SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT],
    )
    models = store.search_logged_models(experiment_ids=[exp_id], page_token=models.token)
    assert_models_match(
        models,
        logged_models[SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT:],
    )
    assert models.token is None


def test_search_logged_models_errors(store):
    exp_id = store.create_experiment("test")
    with pytest.raises(MlflowException, match=r"Invalid attribute key 'artifact_location'"):
        store.search_logged_models(experiment_ids=[exp_id], filter_string="artifact_location='abc'")
    with pytest.raises(
        MlflowException, match=r"`dataset_name` in the `datasets` clause must be specified"
    ):
        store.search_logged_models(experiment_ids=[exp_id], datasets=[{}])
    with pytest.raises(
        MlflowException, match=r"`field_name` in the `order_by` clause must be specified"
    ):
        store.search_logged_models(experiment_ids=[exp_id], order_by=[{}])
    with pytest.raises(MlflowException, match=r"`order_by` must be a list of dictionaries."):
        store.search_logged_models(experiment_ids=[exp_id], order_by=["model_id"])
    with pytest.raises(MlflowException, match=r"Invalid order by field name: artifact_location"):
        store.search_logged_models(
            experiment_ids=[exp_id], order_by=[{"field_name": "artifact_location"}]
        )
    with pytest.raises(MlflowException, match=r"Invalid order by field name: params"):
        store.search_logged_models(experiment_ids=[exp_id], order_by=[{"field_name": "params.x"}])
