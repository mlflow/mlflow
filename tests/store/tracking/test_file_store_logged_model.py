import time
from typing import NamedTuple, Optional
from unittest import mock

import pytest

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT, FileStore


@pytest.fixture
def store(tmp_path):
    return FileStore(str(tmp_path.joinpath("mlruns")))


class CreateLoggedModelArgs(NamedTuple):
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    model_name: Optional[str] = None
    tags: Optional[list[LoggedModelTag]] = None
    params: Optional[list[LoggedModelParameter]] = None
    model_type: Optional[str] = None


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
    assert logged_model.tags == (tags or {})
    assert logged_model.params == (params or {})
    assert logged_model.model_type == model_type
    assert logged_model.status == status


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


def test_create_logged_model_errors(store):
    with pytest.raises(MlflowException, match=r"Could not find experiment with ID 123"):
        store.create_logged_model("123")
    mock_experience = mock.Mock()
    mock_experience.lifecycle_stage = LifecycleStage.DELETED
    with mock.patch(
        "mlflow.store.tracking.file_store.FileStore.get_experiment",
        return_value=mock_experience,
    ):
        with pytest.raises(
            MlflowException,
            match=r"Could not create model under non-active experiment with ID 123",
        ):
            store.create_logged_model("123")

    with pytest.raises(MlflowException, match=r"A key name must be provided."):
        store.create_logged_model(params=[LoggedModelParameter(None, "b")])

    with pytest.raises(MlflowException, match=r"exceeds the maximum length"):
        store.create_logged_model(params=[LoggedModelParameter("a" * 256, "b")])


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


@pytest.mark.parametrize(
    "create_logged_model_args",
    [
        CreateLoggedModelArgs(),
        CreateLoggedModelArgs(experiment_name="test"),
        CreateLoggedModelArgs(experiment_name="test", run_name="test_run"),
        CreateLoggedModelArgs(experiment_name="test", run_name="test_run", model_name="test_model"),
        CreateLoggedModelArgs(
            experiment_name="test",
            run_name="test_run",
            model_name="test_model",
            tags=[LoggedModelTag("tag_key", "tag_value")],
        ),
        CreateLoggedModelArgs(
            experiment_name="test",
            run_name="test_run",
            model_name="test_model",
            tags=[LoggedModelTag("tag_key", "tag_value")],
            params=[LoggedModelParameter("param_key", "param_value")],
        ),
    ],
)
def test_get_logged_model(store, create_logged_model_args):
    experiment_id = (
        store.create_experiment(create_logged_model_args.experiment_name)
        if create_logged_model_args.experiment_name
        else None
    )
    run_id = (
        store.create_run(
            experiment_id,
            user_id="user",
            start_time=0,
            tags=[],
            run_name=create_logged_model_args.run_name,
        ).info.run_id
        if create_logged_model_args.run_name
        else None
    )
    logged_model = store.create_logged_model(
        experiment_id=experiment_id,
        name=create_logged_model_args.model_name,
        source_run_id=run_id,
        tags=create_logged_model_args.tags,
        params=create_logged_model_args.params,
        model_type=create_logged_model_args.model_type,
    )
    fetched_model = store.get_logged_model(logged_model.model_id)
    assert logged_model.model_uri == fetched_model.model_uri
    assert logged_model.to_dictionary() == fetched_model.to_dictionary()


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
    logged_model_dict = logged_model.to_dictionary()
    updated_model = store.finalize_logged_model(logged_model.model_id, LoggedModelStatus.READY)
    assert updated_model.status == str(LoggedModelStatus.READY)
    updated_model_dict = updated_model.to_dictionary()
    for k in logged_model_dict:
        if k not in ["status", "last_updated_timestamp"]:
            assert logged_model_dict[k] == updated_model_dict[k]


def test_finalize_logged_model_errors(store):
    with pytest.raises(MlflowException, match=r"Model '1234' not found"):
        store.finalize_logged_model("1234", LoggedModelStatus.READY)

    logged_model = store.create_logged_model()
    with pytest.raises(MlflowException, match=r"Invalid model status"):
        store.finalize_logged_model(logged_model.model_id, LoggedModelStatus.UNSPECIFIED)


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
        time.sleep(0.1)

    # model_id
    # TODO: do we need to support IN & NOT IN?
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
    assert len(models) == 4

    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"model_id LIKE '{model.model_id}'"
    )
    assert len(models) == 1

    models = store.search_logged_models(
        experiment_ids=[exp_id],
        filter_string=f"model_id ILIKE '{model.model_id.upper()}'",
    )
    assert len(models) == 1

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
    assert len(models) == 4

    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name LIKE '{model.name}'"
    )
    assert len(models) == 1

    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name ILIKE '{model.name.upper()}'"
    )
    assert len(models) == 1

    # model_type
    models = store.search_logged_models(experiment_ids=[exp_id], filter_string="model_type='test'")
    assert len(models) == 5
    models = store.search_logged_models(experiment_ids=[exp_id], filter_string="model_type!='test'")
    assert len(models) == 0
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string="model_type LIKE 'te%'"
    )
    assert len(models) == 5
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string="model_type ILIKE 'TE%'"
    )

    # status
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"status='{LoggedModelStatus.PENDING}'"
    )
    assert len(models) == 5
    store.finalize_logged_model(logged_models[0].model_id, LoggedModelStatus.READY)
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"status!='{LoggedModelStatus.PENDING}'"
    )
    assert len(models) == 1
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"status LIKE '{LoggedModelStatus.READY}'"
    )
    assert len(models) == 1
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string="status ILIKE 'ready'"
    )
    assert len(models) == 1

    # source_run_id
    for run_id in run_ids:
        models = store.search_logged_models(
            experiment_ids=[exp_id], filter_string=f"source_run_id='{run_id}'"
        )
        assert len(models) == 1
    run_id = run_ids[0]
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"source_run_id!='{run_id}'"
    )
    assert len(models) == 4
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"source_run_id LIKE '{run_id}'"
    )
    assert len(models) == 1
    models = store.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"source_run_id ILIKE '{run_id.upper()}'"
    )
    assert len(models) == 1

    # creation_timestamp
    mid_time = logged_models[2].creation_timestamp
    for key in ("creation_timestamp", "creation_time"):
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}={mid_time}"
                )
            )
            == 1
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}<{mid_time}"
                )
            )
            == 2
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}>{mid_time}"
                )
            )
            == 2
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}<={mid_time}"
                )
            )
            == 3
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}>={mid_time}"
                )
            )
            == 3
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}!={mid_time}"
                )
            )
            == 4
        )

    # last_updated_timestamp
    store.set_logged_model_tags(logged_models[0].model_id, [LoggedModelTag("a", "b")])
    max_time = store.get_logged_model(logged_models[0].model_id).last_updated_timestamp
    for key in ("last_updated_timestamp", "last_updated_time"):
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}={max_time}"
                )
            )
            == 1
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}<{max_time}"
                )
            )
            == 4
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}>{max_time}"
                )
            )
            == 0
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}<={max_time}"
                )
            )
            == 5
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}>={max_time}"
                )
            )
            == 1
        )
        assert (
            len(
                store.search_logged_models(
                    experiment_ids=[exp_id], filter_string=f"{key}!={max_time}"
                )
            )
            == 4
        )

    # tags
    assert (
        len(store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a`='b'")) == 1
    )
    store.set_logged_model_tags(logged_models[1].model_id, [LoggedModelTag("a", "b")])
    assert (
        len(store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a`='b'")) == 2
    )
    assert (
        len(store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a`!='b'")) == 0
    )
    assert (
        len(store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a` LIKE 'b'"))
        == 2
    )
    assert (
        len(store.search_logged_models(experiment_ids=[exp_id], filter_string="tags.`a` ILIKE 'B'"))
        == 2
    )


def assert_models_match(models1, models2):
    assert len(models1) == len(models2)
    assert all(
        model1.to_dictionary() == model2.to_dictionary() for model1, model2 in zip(models1, models2)
    )


def test_search_logged_models_order_by(store):
    exp_id = store.create_experiment("test")
    logged_models = []
    for i in range(5):
        run_id = store.create_run(exp_id, "user", 0, [], f"test_run_{i}").info.run_id
        logged_models.append(
            store.create_logged_model(exp_id, source_run_id=run_id, model_type=f"test_{i}")
        )
        # make sure the creation_timestamp is different
        time.sleep(0.1)

    # default: order by creation_timestamp DESC, model_id ASC
    models = store.search_logged_models(experiment_ids=[exp_id])
    assert_models_match(
        models, sorted(logged_models, key=lambda x: (-x.creation_timestamp, x.model_id))
    )

    # model_id
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["model_id ASC"])
    assert_models_match(models, sorted(logged_models, key=lambda x: x.model_id))
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["model_id DESC"])
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: x.model_id, reverse=True),
    )

    # name
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["name"])
    assert_models_match(
        models, sorted(logged_models, key=lambda x: (x.name, -x.creation_timestamp, x.model_id))
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=["name DESC", "model_id DESC"]
    )
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: x.name, reverse=True),
    )

    # model_type
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["model_type"])
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: x.model_type),
    )
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["model_type DESC"])
    assert_models_match(
        models,
        sorted(
            logged_models,
            key=lambda x: x.model_type,
            reverse=True,
        ),
    )

    # status
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["status"])
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (x.status, -x.creation_timestamp, x.model_id)),
    )
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["status DESC"])
    assert_models_match(
        models,
        sorted(
            # all status the same
            logged_models,
            key=lambda x: (x.status, -x.creation_timestamp, x.model_id),
        ),
    )

    # source_run_id
    models = store.search_logged_models(experiment_ids=[exp_id], order_by=["source_run_id"])
    assert_models_match(
        models,
        sorted(logged_models, key=lambda x: (x.source_run_id, -x.creation_timestamp, x.model_id)),
    )
    models = store.search_logged_models(
        experiment_ids=[exp_id], order_by=["source_run_id DESC", "model_id DESC"]
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
    for key in ("creation_timestamp", "creation_time"):
        models = store.search_logged_models(experiment_ids=[exp_id], order_by=[key])
        assert_models_match(
            models,
            sorted(logged_models, key=lambda x: (x.creation_timestamp, x.model_id)),
        )
        models = store.search_logged_models(experiment_ids=[exp_id], order_by=[f"{key} DESC"])
        assert_models_match(
            models,
            sorted(logged_models, key=lambda x: (-x.creation_timestamp, x.model_id)),
        )

    # last_updated_timestamp
    for key in ("last_updated_timestamp", "last_updated_time"):
        models = store.search_logged_models(experiment_ids=[exp_id], order_by=[key])
        assert_models_match(
            models,
            sorted(
                logged_models,
                key=lambda x: (x.last_updated_timestamp, -x.creation_timestamp, x.model_id),
            ),
        )
        models = store.search_logged_models(experiment_ids=[exp_id], order_by=[f"{key} DESC"])
        assert_models_match(
            models,
            sorted(
                logged_models,
                key=lambda x: (-x.last_updated_timestamp, -x.creation_timestamp, x.model_id),
            ),
        )


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
        MlflowException, match=r"Invalid order by key 'artifact_location' specified."
    ):
        store.search_logged_models(experiment_ids=[exp_id], order_by=["artifact_location"])
