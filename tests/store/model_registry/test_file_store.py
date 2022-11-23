import os
import shutil
import time
import unittest

import tempfile
from unittest import mock
import uuid
import pytest

from mlflow.entities.model_registry import (
    ModelVersion,
    RegisteredModelTag,
    ModelVersionTag,
)
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.file_store import FileStore

from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
)
from mlflow.utils.file_utils import write_yaml, path_to_local_file_uri

from tests.helper_functions import random_int, random_str


def now():
    return int(time.time() * 1000)


class TestFileStore(unittest.TestCase):
    ROOT_LOCATION = tempfile.gettempdir()

    def setUp(self):
        self._create_root(TestFileStore.ROOT_LOCATION)

    def get_store(self):
        return FileStore(self.test_root)

    def _create_root(self, root):
        self.test_root = os.path.join(root, f"test_file_store_{random_int()}")
        os.mkdir(self.test_root)

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _create_registered_models_for_test(self):
        self.registered_models = [  # pylint: disable=attribute-defined-outside-init
            random_str() for _ in range(3)
        ]
        self.rm_data = {}  # pylint: disable=attribute-defined-outside-init
        self.mv_data = {}  # pylint: disable=attribute-defined-outside-init
        for name in self.registered_models:
            # create registered model
            creation_time = now()
            rm_folder = os.path.join(self.test_root, FileStore.MODELS_FOLDER_NAME, name)
            os.makedirs(rm_folder)
            d = {
                "name": name,
                "creation_timestamp": creation_time,
                "last_updated_timestamp": creation_time,
                "description": None,
                "latest_versions": [],
                "tags": {},
            }
            self.rm_data[name] = d
            write_yaml(rm_folder, FileStore.META_DATA_FILE_NAME, d)
            # tags
            os.makedirs(os.path.join(rm_folder, FileStore.TAGS_FOLDER_NAME))

    def test_create_registered_model(self):
        fs = self.get_store()

        # Error cases
        with pytest.raises(MlflowException, match="Registered model name cannot be empty."):
            fs.create_registered_model(None)
        with pytest.raises(MlflowException, match="Registered model name cannot be empty."):
            fs.create_registered_model("")

        name = random_str()
        model = fs.create_registered_model(name)
        assert model.name == name
        assert model.latest_versions == []
        assert model.creation_timestamp == model.last_updated_timestamp
        assert model.tags == {}

    def _verify_registered_model(self, fs, name):
        rm = fs.get_registered_model(name)
        assert rm.name == name
        assert rm.creation_timestamp == self.rm_data[name]["creation_timestamp"]
        assert rm.last_updated_timestamp == self.rm_data[name]["last_updated_timestamp"]
        assert rm.description == self.rm_data[name]["description"]
        assert rm.latest_versions == self.rm_data[name]["latest_versions"]
        assert rm.tags == self.rm_data[name]["tags"]

    def test_get_registered_model(self):
        fs = self.get_store()
        self._create_registered_models_for_test()
        for name in self.registered_models:
            self._verify_registered_model(fs, name)

        # test that fake registered models dont exist.
        for name in set(random_str(25) for _ in range(10)):
            with pytest.raises(
                MlflowException, match=f"Could not find registered model with name {name}"
            ):
                fs.get_registered_model(name)

    def test_rename_registered_model(self):
        fs = self.get_store()
        self._create_registered_models_for_test()
        model_name = self.registered_models[random_int(0, len(self.registered_models) - 1)]

        # Error cases
        with pytest.raises(MlflowException, match="Registered model name cannot be empty."):
            fs.rename_registered_model(model_name, None)
        # test that names of existing registered models are checked before renaming
        other_model_name = None
        for name in self.registered_models:
            if name != model_name:
                other_model_name = name
                break
        with pytest.raises(
            MlflowException, match=rf"Registered Model \(name={other_model_name}\) already exists."
        ):
            fs.rename_registered_model(model_name, other_model_name)

        new_name = model_name + "!!!"
        assert model_name != new_name
        fs.rename_registered_model(model_name, new_name)
        assert fs.get_registered_model(new_name).name == new_name

    def _extract_names(self, registered_models):
        return [rm.name for rm in registered_models]

    def test_delete_registered_model(self):
        fs = self.get_store()
        self._create_registered_models_for_test()
        model_name = self.registered_models[random_int(0, len(self.registered_models) - 1)]

        # Error cases
        with pytest.raises(
            MlflowException, match=f"Could not find registered model with name {model_name}!!!"
        ):
            fs.delete_registered_model(model_name + "!!!")

        fs.delete_registered_model(model_name)
        assert model_name not in self._extract_names(
            fs.list_registered_models(max_results=10, page_token=None)
        )

        # Cannot delete a deleted model
        with pytest.raises(
            MlflowException, match=f"Could not find registered model with name {model_name}"
        ):
            fs.delete_registered_model(model_name)

    def test_list_registered_model(self):
        fs = self.get_store()
        self._create_registered_models_for_test()
        for rm in fs.list_registered_models(max_results=10, page_token=None):
            name = rm.name
            assert name in self.registered_models
            assert name == self.rm_data[name]["name"]

    def test_list_registered_model_paginated(self):
        fs = self.get_store()
        for _ in range(10):
            fs.create_registered_model(random_str())
        rms1 = fs.list_registered_models(max_results=4, page_token=None)
        assert len(rms1) == 4
        assert rms1.token is not None
        rms2 = fs.list_registered_models(max_results=4, page_token=None)
        assert len(rms2) == 4
        assert rms2.token is not None
        assert rms1 == rms2
        rms3 = fs.list_registered_models(max_results=500, page_token=rms2.token)
        assert len(rms3) <= 500
        if len(rms3) < 500:
            assert rms3.token is None

    def test_list_registered_model_paginated_returns_in_correct_order(self):
        fs = self.get_store()

        rms = [fs.create_registered_model("RM{:03}".format(i)).name for i in range(50)]

        # test that pagination will return all valid results in sorted order
        # by name ascending
        result = fs.list_registered_models(max_results=5, page_token=None)
        assert result.token is not None
        assert self._extract_names(result) == rms[0:5]

        result = fs.list_registered_models(page_token=result.token, max_results=10)
        assert result.token is not None
        assert self._extract_names(result) == rms[5:15]

        result = fs.list_registered_models(page_token=result.token, max_results=20)
        assert result.token is not None
        assert self._extract_names(result) == rms[15:35]

        result = fs.list_registered_models(page_token=result.token, max_results=100)
        assert result.token is None
        assert self._extract_names(result) == rms[35:]

    def test_list_registered_model_paginated_errors(self):
        fs = self.get_store()
        rms = [fs.create_registered_model("RM{:03}".format(i)).name for i in range(50)]
        # test that providing a completely invalid page token throws
        with pytest.raises(
            MlflowException, match=r"Invalid page token, could not base64-decode"
        ) as exception_context:
            fs.list_registered_models(page_token="evilhax", max_results=20)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # test that providing too large of a max_results throws
        with pytest.raises(
            MlflowException, match=r"Invalid value for max_results"
        ) as exception_context:
            fs.list_registered_models(page_token="evilhax", max_results=1e15)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # list should not return deleted models
        fs.delete_registered_model(name="RM{0:03}".format(0))
        assert set(
            self._extract_names(fs.list_registered_models(max_results=100, page_token=None))
        ) == set(rms[1:])

    def _extract_latest_by_stage(self, latest_versions):
        return {mvd.current_stage: mvd.version for mvd in latest_versions}

    def _create_model_version(
        self,
        fs,
        name,
        source="path/to/source",
        run_id=uuid.uuid4().hex,
        tags=None,
        run_link=None,
        description=None,
    ):
        return fs.create_model_version(
            name, source, run_id, tags, run_link=run_link, description=description
        )

    def test_get_latest_versions(self):
        fs = self.get_store()
        name = "test_for_latest_versions"
        rmd1 = fs.create_registered_model(name)
        assert rmd1.latest_versions == []

        mv1 = self._create_model_version(fs, name)
        assert mv1.version == 1
        rmd2 = fs.get_registered_model(name)
        assert self._extract_latest_by_stage(rmd2.latest_versions) == {"None": 1}

        # add a bunch more
        mv2 = self._create_model_version(fs, name)
        assert mv2.version == 2
        fs.transition_model_version_stage(
            name=mv2.name, version=mv2.version, stage="Production", archive_existing_versions=False
        )

        mv3 = self._create_model_version(fs, name)
        assert mv3.version == 3
        fs.transition_model_version_stage(
            name=mv3.name, version=mv3.version, stage="Production", archive_existing_versions=False
        )
        mv4 = self._create_model_version(fs, name)
        assert mv4.version == 4
        fs.transition_model_version_stage(
            name=mv4.name, version=mv4.version, stage="Staging", archive_existing_versions=False
        )

        # test that correct latest versions are returned for each stage
        rmd4 = fs.get_registered_model(name)
        assert self._extract_latest_by_stage(rmd4.latest_versions) == {
            "None": 1,
            "Production": 3,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(fs.get_latest_versions(name=name, stages=None)) == {
            "None": 1,
            "Production": 3,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(fs.get_latest_versions(name=name, stages=[])) == {
            "None": 1,
            "Production": 3,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(
            fs.get_latest_versions(name=name, stages=["Production"])
        ) == {"Production": 3}
        assert self._extract_latest_by_stage(
            fs.get_latest_versions(name=name, stages=["production"])
        ) == {
            "Production": 3
        }  # The stages are case insensitive.
        assert self._extract_latest_by_stage(
            fs.get_latest_versions(name=name, stages=["pROduction"])
        ) == {
            "Production": 3
        }  # The stages are case insensitive.
        assert self._extract_latest_by_stage(
            fs.get_latest_versions(name=name, stages=["None", "Production"])
        ) == {"None": 1, "Production": 3}

        # delete latest Production, and should point to previous one
        fs.delete_model_version(name=mv3.name, version=mv3.version)
        rmd5 = fs.get_registered_model(name=name)
        assert self._extract_latest_by_stage(rmd5.latest_versions) == {
            "None": 1,
            "Production": 2,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(fs.get_latest_versions(name=name, stages=None)) == {
            "None": 1,
            "Production": 2,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(
            fs.get_latest_versions(name=name, stages=["Production"])
        ) == {"Production": 2}

    def test_set_registered_model_tag(self):
        fs = self.get_store()
        name1 = "SetRegisteredModelTag_TestMod"
        name2 = "SetRegisteredModelTag_TestMod 2"
        initial_tags = [
            RegisteredModelTag("key", "value"),
            RegisteredModelTag("anotherKey", "some other value"),
        ]
        fs.create_registered_model(name1, initial_tags)
        fs.create_registered_model(name2, initial_tags)
        new_tag = RegisteredModelTag("randomTag", "not a random value")
        fs.set_registered_model_tag(name1, new_tag)
        rm1 = fs.get_registered_model(name=name1)
        all_tags = initial_tags + [new_tag]
        assert rm1.tags == {tag.key: tag.value for tag in all_tags}

        # test overriding a tag with the same key
        overriding_tag = RegisteredModelTag("key", "overriding")
        fs.set_registered_model_tag(name1, overriding_tag)
        all_tags = [tag for tag in all_tags if tag.key != "key"] + [overriding_tag]
        rm1 = fs.get_registered_model(name=name1)
        assert rm1.tags == {tag.key: tag.value for tag in all_tags}
        # does not affect other models with the same key
        rm2 = fs.get_registered_model(name=name2)
        assert rm2.tags == {tag.key: tag.value for tag in initial_tags}

        # can not set tag on deleted (non-existed) registered model
        fs.delete_registered_model(name1)
        with pytest.raises(
            MlflowException, match=f"Could not find registered model with name {name1}"
        ) as exception_context:
            fs.set_registered_model_tag(name1, overriding_tag)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # test cannot set tags that are too long
        long_tag = RegisteredModelTag("longTagKey", "a" * 5001)
        with pytest.raises(
            MlflowException,
            match=(
                r"Registered model value '.+' had length \d+, which exceeded length limit of 5000"
            ),
        ) as exception_context:
            fs.set_registered_model_tag(name2, long_tag)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # test can set tags that are somewhat long
        long_tag = RegisteredModelTag("longTagKey", "a" * 4999)
        fs.set_registered_model_tag(name2, long_tag)
        # can not set invalid tag
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            fs.set_registered_model_tag(name2, RegisteredModelTag(key=None, value=""))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            fs.set_registered_model_tag(None, RegisteredModelTag(key="key", value="value"))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_delete_registered_model_tag(self):
        fs = self.get_store()
        name1 = "DeleteRegisteredModelTag_TestMod"
        name2 = "DeleteRegisteredModelTag_TestMod 2"
        initial_tags = [
            RegisteredModelTag("key", "value"),
            RegisteredModelTag("anotherKey", "some other value"),
        ]
        fs.create_registered_model(name1, initial_tags)
        fs.create_registered_model(name2, initial_tags)
        new_tag = RegisteredModelTag("randomTag", "not a random value")
        fs.set_registered_model_tag(name1, new_tag)
        fs.delete_registered_model_tag(name1, "randomTag")
        rm1 = fs.get_registered_model(name=name1)
        assert rm1.tags == {tag.key: tag.value for tag in initial_tags}

        # testing deleting a key does not affect other models with the same key
        fs.delete_registered_model_tag(name1, "key")
        rm1 = fs.get_registered_model(name=name1)
        rm2 = fs.get_registered_model(name=name2)
        assert rm1.tags == {"anotherKey": "some other value"}
        assert rm2.tags == {tag.key: tag.value for tag in initial_tags}

        # delete tag that is already deleted does nothing
        fs.delete_registered_model_tag(name1, "key")
        rm1 = fs.get_registered_model(name=name1)
        assert rm1.tags == {"anotherKey": "some other value"}

        # can not delete tag on deleted (non-existed) registered model
        fs.delete_registered_model(name1)
        with pytest.raises(
            MlflowException, match=f"Could not find registered model with name {name1}"
        ) as exception_context:
            fs.delete_registered_model_tag(name1, "anotherKey")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # can not delete tag with invalid key
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            fs.delete_registered_model_tag(name2, None)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            fs.delete_registered_model_tag(None, "key")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_create_model_version(self):
        fs = self.get_store()
        name = "test_for_create_MV"
        fs.create_registered_model(name)
        run_id = uuid.uuid4().hex
        with mock.patch("time.time") as mock_time:
            mock_time.return_value = 456778
            mv1 = fs.create_model_version(name, "a/b/CD", run_id)
            assert mv1.name == name
            assert mv1.version == 1

        mvd1 = fs.get_model_version(mv1.name, mv1.version)
        assert mvd1.name == name
        assert mvd1.version == 1
        assert mvd1.current_stage == "None"
        assert mvd1.creation_timestamp == 456778000
        assert mvd1.last_updated_timestamp == 456778000
        assert mvd1.description is None
        assert mvd1.source == "a/b/CD"
        assert mvd1.run_id == run_id
        assert mvd1.status == "READY"
        assert mvd1.status_message is None
        assert mvd1.tags == {}

        # new model versions for same name autoincrement versions
        mv2 = self._create_model_version(fs, name)
        mvd2 = fs.get_model_version(name=mv2.name, version=mv2.version)
        assert mv2.version == 2
        assert mvd2.version == 2

        # create model version with tags return model version entity with tags
        tags = [ModelVersionTag("key", "value"), ModelVersionTag("anotherKey", "some other value")]
        mv3 = self._create_model_version(fs, name, tags=tags)
        mvd3 = fs.get_model_version(name=mv3.name, version=mv3.version)
        assert mv3.version == 3
        assert mv3.tags == {tag.key: tag.value for tag in tags}
        assert mvd3.version == 3
        assert mvd3.tags == {tag.key: tag.value for tag in tags}

        # create model versions with runLink
        run_link = "http://localhost:3000/path/to/run/"
        mv4 = self._create_model_version(fs, name, run_link=run_link)
        mvd4 = fs.get_model_version(name, mv4.version)
        assert mv4.version == 4
        assert mv4.run_link == run_link
        assert mvd4.version == 4
        assert mvd4.run_link == run_link

        # create model version with description
        description = "the best model ever"
        mv5 = self._create_model_version(fs, name, description=description)
        mvd5 = fs.get_model_version(name, mv5.version)
        assert mv5.version == 5
        assert mv5.description == description
        assert mvd5.version == 5
        assert mvd5.description == description

        # create model version without runId
        mv6 = self._create_model_version(fs, name, run_id=None)
        mvd6 = fs.get_model_version(name, mv6.version)
        assert mv6.version == 6
        assert mv6.run_id is None
        assert mvd6.version == 6
        assert mvd6.run_id is None

    def test_update_model_version(self):
        fs = self.get_store()
        name = "test_for_update_MV"
        fs.create_registered_model(name)
        mv1 = self._create_model_version(fs, name)
        mvd1 = fs.get_model_version(name=mv1.name, version=mv1.version)
        assert mvd1.name == name
        assert mvd1.version == 1
        assert mvd1.current_stage == "None"

        # update stage
        fs.transition_model_version_stage(
            name=mv1.name, version=mv1.version, stage="Production", archive_existing_versions=False
        )
        mvd2 = fs.get_model_version(name=mv1.name, version=mv1.version)
        assert mvd2.name == name
        assert mvd2.version == 1
        assert mvd2.current_stage == "Production"
        assert mvd2.description is None

        # update description
        fs.update_model_version(
            name=mv1.name, version=mv1.version, description="test model version"
        )
        mvd3 = fs.get_model_version(name=mv1.name, version=mv1.version)
        assert mvd3.name == name
        assert mvd3.version == 1
        assert mvd3.current_stage == "Production"
        assert mvd3.description == "test model version"

        # only valid stages can be set
        with pytest.raises(
            MlflowException,
            match=(
                "Invalid Model Version stage: unknown. "
                "Value must be one of None, Staging, Production, Archived."
            ),
        ) as exception_context:
            fs.transition_model_version_stage(
                mv1.name, mv1.version, stage="unknown", archive_existing_versions=False
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # stages are case-insensitive and auto-corrected to system stage names
        for stage_name in ["STAGING", "staging", "StAgInG"]:
            fs.transition_model_version_stage(
                name=mv1.name,
                version=mv1.version,
                stage=stage_name,
                archive_existing_versions=False,
            )
            mvd5 = fs.get_model_version(name=mv1.name, version=mv1.version)
            assert mvd5.current_stage == "Staging"

    def test_transition_model_version_stage_when_archive_existing_versions_is_false(self):
        fs = self.get_store()
        name = "model"
        fs.create_registered_model(name)
        mv1 = self._create_model_version(fs, name)
        mv2 = self._create_model_version(fs, name)
        mv3 = self._create_model_version(fs, name)

        # test that when `archive_existing_versions` is False, transitioning a model version
        # to the inactive stages ("Archived" and "None") does not throw.
        for stage in ["Archived", "None"]:
            fs.transition_model_version_stage(name, mv1.version, stage, False)

        fs.transition_model_version_stage(name, mv1.version, "Staging", False)
        fs.transition_model_version_stage(name, mv2.version, "Production", False)
        fs.transition_model_version_stage(name, mv3.version, "Staging", False)

        mvd1 = fs.get_model_version(name=name, version=mv1.version)
        mvd2 = fs.get_model_version(name=name, version=mv2.version)
        mvd3 = fs.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Staging"
        assert mvd2.current_stage == "Production"
        assert mvd3.current_stage == "Staging"

        fs.transition_model_version_stage(name, mv3.version, "Production", False)

        mvd1 = fs.get_model_version(name=name, version=mv1.version)
        mvd2 = fs.get_model_version(name=name, version=mv2.version)
        mvd3 = fs.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Staging"
        assert mvd2.current_stage == "Production"
        assert mvd3.current_stage == "Production"

    def test_transition_model_version_stage_when_archive_existing_versions_is_true(self):
        fs = self.get_store()
        name = "model"
        fs.create_registered_model(name)
        mv1 = self._create_model_version(fs, name)
        mv2 = self._create_model_version(fs, name)
        mv3 = self._create_model_version(fs, name)

        msg = (
            r"Model version transition cannot archive existing model versions "
            r"because .+ is not an Active stage"
        )

        # test that when `archive_existing_versions` is True, transitioning a model version
        # to the inactive stages ("Archived" and "None") throws.
        for stage in ["Archived", "None"]:
            with pytest.raises(MlflowException, match=msg):
                fs.transition_model_version_stage(name, mv1.version, stage, True)

        fs.transition_model_version_stage(name, mv1.version, "Staging", False)
        fs.transition_model_version_stage(name, mv2.version, "Production", False)
        fs.transition_model_version_stage(name, mv3.version, "Staging", True)

        mvd1 = fs.get_model_version(name=name, version=mv1.version)
        mvd2 = fs.get_model_version(name=name, version=mv2.version)
        mvd3 = fs.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Archived"
        assert mvd2.current_stage == "Production"
        assert mvd3.current_stage == "Staging"
        assert mvd1.last_updated_timestamp == mvd3.last_updated_timestamp

        fs.transition_model_version_stage(name, mv3.version, "Production", True)

        mvd1 = fs.get_model_version(name=name, version=mv1.version)
        mvd2 = fs.get_model_version(name=name, version=mv2.version)
        mvd3 = fs.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Archived"
        assert mvd2.current_stage == "Archived"
        assert mvd3.current_stage == "Production"
        assert mvd2.last_updated_timestamp == mvd3.last_updated_timestamp

        for uncanonical_stage_name in ["STAGING", "staging", "StAgInG"]:
            fs.transition_model_version_stage(mv1.name, mv1.version, "Staging", False)
            fs.transition_model_version_stage(mv2.name, mv2.version, "None", False)

            # stage names are case-insensitive and auto-corrected to system stage names
            fs.transition_model_version_stage(mv2.name, mv2.version, uncanonical_stage_name, True)

            mvd1 = fs.get_model_version(name=mv1.name, version=mv1.version)
            mvd2 = fs.get_model_version(name=mv2.name, version=mv2.version)
            assert mvd1.current_stage == "Archived"
            assert mvd2.current_stage == "Staging"

    def test_delete_model_version(self):
        fs = self.get_store()
        name = "test_for_delete_MV"
        initial_tags = [
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "some other value"),
        ]
        fs.create_registered_model(name)
        mv = self._create_model_version(fs, name, tags=initial_tags)
        mvd = fs.get_model_version(name=mv.name, version=mv.version)
        assert mvd.name == name

        fs.delete_model_version(name=mv.name, version=mv.version)

        # cannot get a deleted model version
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            fs.get_model_version(name=mv.name, version=mv.version)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot update a delete
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            fs.update_model_version(mv.name, mv.version, description="deleted!")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot delete it again
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            fs.delete_model_version(name=mv.name, version=mv.version)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_search_model_versions(self):
        fs = self.get_store()
        # create some model versions
        name = "test_for_search_MV"
        fs.create_registered_model(name)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        mv1 = self._create_model_version(fs, name=name, source="A/B", run_id=run_id_1)
        assert mv1.version == 1
        mv2 = self._create_model_version(fs, name=name, source="A/C", run_id=run_id_2)
        assert mv2.version == 2
        mv3 = self._create_model_version(fs, name=name, source="A/D", run_id=run_id_2)
        assert mv3.version == 3
        mv4 = self._create_model_version(fs, name=name, source="A/D", run_id=run_id_3)
        assert mv4.version == 4

        def search_versions(filter_string):
            return [mvd.version for mvd in fs.search_model_versions(filter_string)]

        # search using name should return all 4 versions
        assert set(search_versions("name='%s'" % name)) == {1, 2, 3, 4}

        # search using run_id_1 should return version 1
        assert set(search_versions("run_id='%s'" % run_id_1)) == {1}

        # search using run_id_2 should return versions 2 and 3
        assert set(search_versions("run_id='%s'" % run_id_2)) == {2, 3}

        # search using the IN operator should return all versions
        assert set(search_versions(f"run_id IN ('{run_id_1}','{run_id_2}')")) == {1, 2, 3}

        # search IN operator is case sensitive
        assert set(search_versions(f"run_id IN ('{run_id_1.upper()}','{run_id_2}')")) == {2, 3}

        # search IN operator with right-hand side value containing whitespaces
        assert set(search_versions(f"run_id IN ('{run_id_1}', '{run_id_2}')")) == {1, 2, 3}

        # search using the IN operator with bad lists should return exceptions
        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected string value, punctuation, or whitespace, "
                r"but got different type in list"
            ),
        ) as exception_context:
            search_versions("run_id IN (1,2,3)")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        assert set(search_versions(f"run_id LIKE '{run_id_2[:30]}%'")) == {2, 3}

        assert set(search_versions(f"run_id ILIKE '{run_id_2[:30].upper()}%'")) == {2, 3}

        # search using the IN operator with empty lists should return exceptions
        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected a non-empty list of string values, "
                r"but got empty list"
            ),
        ) as exception_context:
            search_versions("run_id IN ()")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # search using an ill-formed IN operator correctly throws exception
        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            search_versions("run_id IN (")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            search_versions("run_id IN")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            search_versions("name LIKE")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected a non-empty list of string values, "
                r"but got ill-formed list"
            ),
        ) as exception_context:
            search_versions("run_id IN (,)")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected a non-empty list of string values, "
                r"but got ill-formed list"
            ),
        ) as exception_context:
            search_versions("run_id IN ('runid1',,'runid2')")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # delete mv4. search should not return version 4
        fs.delete_model_version(name=mv4.name, version=mv4.version)
        assert set(search_versions("")) == {1, 2, 3}

        assert set(search_versions(None)) == {1, 2, 3}

        assert set(search_versions("name='%s'" % name)) == {1, 2, 3}

        fs.transition_model_version_stage(
            name=mv1.name, version=mv1.version, stage="production", archive_existing_versions=False
        )

        fs.update_model_version(
            name=mv1.name, version=mv1.version, description="Online prediction model!"
        )

        mvds = fs.search_model_versions("run_id = '%s'" % run_id_1)
        assert 1 == len(mvds)
        assert isinstance(mvds[0], ModelVersion)
        assert mvds[0].current_stage == "Production"
        assert mvds[0].run_id == run_id_1
        assert mvds[0].source == "A/B"
        assert mvds[0].description == "Online prediction model!"

    def test_search_model_versions_by_tag(self):
        fs = self.get_store()
        # create some model versions
        name = "test_for_search_MV_by_tag"
        fs.create_registered_model(name)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex

        mv1 = self._create_model_version(
            fs,
            name=name,
            source="A/B",
            run_id=run_id_1,
            tags=[ModelVersionTag("t1", "abc"), ModelVersionTag("t2", "xyz")],
        )
        assert mv1.version == 1
        mv2 = self._create_model_version(
            fs,
            name=name,
            source="A/C",
            run_id=run_id_2,
            tags=[ModelVersionTag("t1", "abc"), ModelVersionTag("t2", "x123")],
        )
        assert mv2.version == 2

        def search_versions(filter_string):
            return [mvd.version for mvd in fs.search_model_versions(filter_string)]

        assert search_versions(f"name = '{name}' and tag.t2 = 'xyz'") == [1]
        assert search_versions("name = 'wrong_name' and tag.t2 = 'xyz'") == []
        assert search_versions("tag.`t2` = 'xyz'") == [1]
        assert search_versions("tag.t3 = 'xyz'") == []
        assert set(search_versions("tag.t2 != 'xy'")) == {2, 1}
        assert search_versions("tag.t2 LIKE 'xy%'") == [1]
        assert search_versions("tag.t2 LIKE 'xY%'") == []
        assert search_versions("tag.t2 ILIKE 'xY%'") == [1]
        assert set(search_versions("tag.t2 LIKE 'x%'")) == {2, 1}
        assert search_versions("tag.T2 = 'xyz'") == []
        assert search_versions("tag.t1 = 'abc' and tag.t2 = 'xyz'") == [1]
        assert set(search_versions("tag.t1 = 'abc' and tag.t2 LIKE 'x%'")) == {2, 1}
        assert search_versions("tag.t1 = 'abc' and tag.t2 LIKE 'y%'") == []
        # test filter with duplicated keys
        assert search_versions("tag.t2 like 'x%' and tag.t2 != 'xyz'") == [2]

    def _search_registered_models(
        self, fs, filter_string=None, max_results=10, order_by=None, page_token=None
    ):
        result = fs.search_registered_models(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        return [registered_model.name for registered_model in result], result.token

    def test_search_registered_models(self):
        fs = self.get_store()
        # create some registered models
        prefix = "test_for_search_"
        names = [prefix + name for name in ["RM1", "RM2", "RM3", "RM4", "RM4A", "RM4ab"]]
        for name in names:
            fs.create_registered_model(name)

        # search with no filter should return all registered models
        rms, _ = self._search_registered_models(fs, None)
        assert rms == names

        # equality search using name should return exactly the 1 name
        rms, _ = self._search_registered_models(fs, "name='{}'".format(names[0]))
        assert rms == [names[0]]

        # equality search using name that is not valid should return nothing
        rms, _ = self._search_registered_models(fs, "name='{}'".format(names[0] + "cats"))
        assert rms == []

        # case-sensitive prefix search using LIKE should return all the RMs
        rms, _ = self._search_registered_models(fs, "name LIKE '{}%'".format(prefix))
        assert rms == names

        # case-sensitive prefix search using LIKE with surrounding % should return all the RMs
        rms, _ = self._search_registered_models(fs, "name LIKE '%RM%'")
        assert rms == names

        # case-sensitive prefix search using LIKE with surrounding % should return all the RMs
        # _e% matches test_for_search_ , so all RMs should match
        rms, _ = self._search_registered_models(fs, "name LIKE '_e%'")
        assert rms == names

        # case-sensitive prefix search using LIKE should return just rm4
        rms, _ = self._search_registered_models(fs, "name LIKE '{}%'".format(prefix + "RM4A"))
        assert rms == [names[4]]

        # case-sensitive prefix search using LIKE should return no models if no match
        rms, _ = self._search_registered_models(fs, "name LIKE '{}%'".format(prefix + "cats"))
        assert rms == []

        # confirm that LIKE is not case-sensitive
        rms, _ = self._search_registered_models(fs, "name lIkE '%blah%'")
        assert rms == []

        rms, _ = self._search_registered_models(fs, "name like '{}%'".format(prefix + "RM4A"))
        assert rms == [names[4]]

        # case-insensitive prefix search using ILIKE should return both rm5 and rm6
        rms, _ = self._search_registered_models(fs, "name ILIKE '{}%'".format(prefix + "RM4A"))
        assert rms == names[4:]

        # case-insensitive postfix search with ILIKE
        rms, _ = self._search_registered_models(fs, "name ILIKE '%RM4a%'")
        assert rms == names[4:]

        # case-insensitive prefix search using ILIKE should return both rm5 and rm6
        rms, _ = self._search_registered_models(fs, "name ILIKE '{}%'".format(prefix + "cats"))
        assert rms == []

        # confirm that ILIKE is not case-sensitive
        rms, _ = self._search_registered_models(fs, "name iLike '%blah%'")
        assert rms == []

        # confirm that ILIKE works for empty query
        rms, _ = self._search_registered_models(fs, "name iLike '%%'")
        assert rms == names

        rms, _ = self._search_registered_models(fs, "name ilike '%RM4a%'")
        assert rms == names[4:]

        # cannot search by invalid comparator types
        with pytest.raises(
            MlflowException,
            match="Parameter value is either not quoted or unidentified quote types used for "
            "string value something",
        ) as exception_context:
            self._search_registered_models(fs, "name!=something")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # cannot search by run_id
        with pytest.raises(
            MlflowException,
            match=r"Invalid attribute key 'run_id' specified. Valid keys are '{'name'}'",
        ) as exception_context:
            self._search_registered_models(fs, "run_id='%s'" % "somerunID")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # cannot search by source_path
        with pytest.raises(
            MlflowException,
            match=r"Invalid attribute key 'source_path' specified. Valid keys are '{'name'}'",
        ) as exception_context:
            self._search_registered_models(fs, "source_path = 'A/D'")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # cannot search by other params
        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            self._search_registered_models(fs, "evilhax = true")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # delete last registered model. search should not return the first 5
        fs.delete_registered_model(name=names[-1])
        assert self._search_registered_models(fs, None, max_results=1000) == (names[:-1], None)

        # equality search using name should return no names
        assert self._search_registered_models(fs, "name='{}'".format(names[-1])) == ([], None)

        # case-sensitive prefix search using LIKE should return all the RMs
        assert self._search_registered_models(fs, "name LIKE '{}%'".format(prefix)) == (
            names[0:5],
            None,
        )

        # case-insensitive prefix search using ILIKE should return both rm5 and rm6
        assert self._search_registered_models(fs, "name ILIKE '{}%'".format(prefix + "RM4A")) == (
            [names[4]],
            None,
        )

    def test_search_registered_models_by_tag(self):
        fs = self.get_store()
        name1 = "test_for_search_RM_by_tag1"
        name2 = "test_for_search_RM_by_tag2"
        tags1 = [
            RegisteredModelTag("t1", "abc"),
            RegisteredModelTag("t2", "xyz"),
        ]
        tags2 = [
            RegisteredModelTag("t1", "abcd"),
            RegisteredModelTag("t2", "xyz123"),
            RegisteredModelTag("t3", "XYZ"),
        ]
        fs.create_registered_model(name1, tags1)
        fs.create_registered_model(name2, tags2)

        rms, _ = self._search_registered_models(fs, "tag.t3 = 'XYZ'")
        assert rms == [name2]

        rms, _ = self._search_registered_models(fs, f"name = '{name1}' and tag.t1 = 'abc'")
        assert rms == [name1]

        rms, _ = self._search_registered_models(fs, "tag.t1 LIKE 'ab%'")
        assert rms == [name1, name2]

        rms, _ = self._search_registered_models(fs, "tag.t1 ILIKE 'aB%'")
        assert rms == [name1, name2]

        rms, _ = self._search_registered_models(fs, "tag.t1 LIKE 'ab%' AND tag.t2 LIKE 'xy%'")
        assert rms == [name1, name2]

        rms, _ = self._search_registered_models(fs, "tag.t3 = 'XYz'")
        assert rms == []

        rms, _ = self._search_registered_models(fs, "tag.T3 = 'XYZ'")
        assert rms == []

        rms, _ = self._search_registered_models(fs, "tag.t1 != 'abc'")
        assert rms == [name2]

        # test filter with duplicated keys
        rms, _ = self._search_registered_models(fs, "tag.t1 != 'abcd' and tag.t1 LIKE 'ab%'")
        assert rms == [name1]

    def test_search_registered_models_order_by_simple(self):
        fs = self.get_store()
        # create some registered models
        prefix = "test_for_search_"
        names = [prefix + name for name in ["RM1", "RM2", "RM3", "RM4", "RM4A", "RM4ab"]]
        for name in names:
            fs.create_registered_model(name)
            time.sleep(0.001)  # sleep for windows fs timestamp precision issues

        # by default order by name ASC
        rms, _ = self._search_registered_models(fs)
        assert rms == names

        # order by name DESC
        rms, _ = self._search_registered_models(fs, order_by=["name DESC"])
        assert rms == names[::-1]

        # order by last_updated_timestamp ASC
        fs.update_registered_model(names[0], "latest updated")
        rms, _ = self._search_registered_models(fs, order_by=["last_updated_timestamp ASC"])
        assert rms[-1] == names[0]

    def test_search_registered_model_pagination(self):
        fs = self.get_store()
        rms = [fs.create_registered_model("RM{:03}".format(i)).name for i in range(50)]

        # test flow with fixed max_results
        returned_rms = []
        query = "name LIKE 'RM%'"
        result, token = self._search_registered_models(fs, query, page_token=None, max_results=5)
        returned_rms.extend(result)
        while token:
            result, token = self._search_registered_models(
                fs, query, page_token=token, max_results=5
            )
            returned_rms.extend(result)
        assert rms == returned_rms

        # test that pagination will return all valid results in sorted order
        # by name ascending
        result, token1 = self._search_registered_models(fs, query, max_results=5)
        assert token1 is not None
        assert result == rms[0:5]

        result, token2 = self._search_registered_models(
            fs, query, page_token=token1, max_results=10
        )
        assert token2 is not None
        assert result == rms[5:15]

        result, token3 = self._search_registered_models(
            fs, query, page_token=token2, max_results=20
        )
        assert token3 is not None
        assert result == rms[15:35]

        result, token4 = self._search_registered_models(
            fs, query, page_token=token3, max_results=100
        )
        # assert that page token is None
        assert token4 is None
        assert result == rms[35:]

        # test that providing a completely invalid page token throws
        with pytest.raises(
            MlflowException, match=r"Invalid page token, could not base64-decode"
        ) as exception_context:
            self._search_registered_models(fs, query, page_token="evilhax", max_results=20)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # test that providing too large of a max_results throws
        with pytest.raises(
            MlflowException, match=r"Invalid value for max_results."
        ) as exception_context:
            self._search_registered_models(fs, query, page_token="evilhax", max_results=1e15)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_search_registered_model_order_by(self):
        fs = self.get_store()
        rms = []
        # explicitly mock the creation_timestamps because timestamps seem to be unstable in Windows
        for i in range(50):
            rms.append(fs.create_registered_model("RM{:03}".format(i)).name)
            time.sleep(0.01)

        # test flow with fixed max_results and order_by (test stable order across pages)
        returned_rms = []
        query = "name LIKE 'RM%'"
        result, token = self._search_registered_models(
            fs, query, page_token=None, order_by=["name DESC"], max_results=5
        )
        returned_rms.extend(result)
        while token:
            result, token = self._search_registered_models(
                fs, query, page_token=token, order_by=["name DESC"], max_results=5
            )
            returned_rms.extend(result)
        # name descending should be the opposite order of the current order
        assert rms[::-1] == returned_rms
        # last_updated_timestamp descending should have the newest RMs first
        result, _ = self._search_registered_models(
            fs, query, page_token=None, order_by=["last_updated_timestamp DESC"], max_results=100
        )
        assert rms[::-1] == result
        # last_updated_timestamp ascending should have the oldest RMs first
        result, _ = self._search_registered_models(
            fs, query, page_token=None, order_by=["last_updated_timestamp ASC"], max_results=100
        )
        assert rms == result
        # name ascending should have the original order
        result, _ = self._search_registered_models(
            fs, query, page_token=None, order_by=["name ASC"], max_results=100
        )
        assert rms == result
        # test that no ASC/DESC defaults to ASC
        result, _ = self._search_registered_models(
            fs, query, page_token=None, order_by=["last_updated_timestamp"], max_results=100
        )
        assert rms == result
        with mock.patch("mlflow.store.model_registry.file_store.now", return_value=1):
            rm1 = fs.create_registered_model("MR1").name
            rm2 = fs.create_registered_model("MR2").name
        with mock.patch("mlflow.store.model_registry.file_store.now", return_value=2):
            rm3 = fs.create_registered_model("MR3").name
            rm4 = fs.create_registered_model("MR4").name
        query = "name LIKE 'MR%'"
        # test with multiple clauses
        result, _ = self._search_registered_models(
            fs,
            query,
            page_token=None,
            order_by=["last_updated_timestamp ASC", "name DESC"],
            max_results=100,
        )
        assert [rm2, rm1, rm4, rm3] == result
        # confirm that name ascending is the default, even if ties exist on other fields
        result, _ = self._search_registered_models(
            fs, query, page_token=None, order_by=[], max_results=100
        )
        assert [rm1, rm2, rm3, rm4] == result
        # test default tiebreak with descending timestamps
        result, _ = self._search_registered_models(
            fs, query, page_token=None, order_by=["last_updated_timestamp DESC"], max_results=100
        )
        assert [rm3, rm4, rm1, rm2] == result

    def test_search_registered_model_order_by_errors(self):
        fs = self.get_store()
        fs.create_registered_model("dummy")
        query = "name LIKE 'RM%'"
        # test that invalid columns throw even if they come after valid columns
        with pytest.raises(
            MlflowException, match="Invalid attribute key 'description' specified."
        ) as exception_context:
            self._search_registered_models(
                fs,
                query,
                page_token=None,
                order_by=["name ASC", "description DESC"],
                max_results=5,
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # test that invalid columns with random text throw even if they come after valid columns
        with pytest.raises(
            MlflowException, match=r"Invalid order_by clause '.+'"
        ) as exception_context:
            self._search_registered_models(
                fs,
                query,
                page_token=None,
                order_by=["name ASC", "last_updated_timestamp DESC blah"],
                max_results=5,
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_set_model_version_tag(self):
        fs = self.get_store()
        name1 = "SetModelVersionTag_TestMod"
        name2 = "SetModelVersionTag_TestMod 2"
        initial_tags = [
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "some other value"),
        ]
        fs.create_registered_model(name1)
        fs.create_registered_model(name2)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        fs.create_model_version(name1, "A/B", run_id_1, initial_tags)
        fs.create_model_version(name1, "A/C", run_id_2, initial_tags)
        fs.create_model_version(name2, "A/D", run_id_3, initial_tags)
        new_tag = ModelVersionTag("randomTag", "not a random value")
        fs.set_model_version_tag(name1, 1, new_tag)
        all_tags = initial_tags + [new_tag]
        rm1mv1 = fs.get_model_version(name1, 1)
        assert rm1mv1.tags == {tag.key: tag.value for tag in all_tags}

        # test overriding a tag with the same key
        overriding_tag = ModelVersionTag("key", "overriding")
        fs.set_model_version_tag(name1, 1, overriding_tag)
        all_tags = [tag for tag in all_tags if tag.key != "key"] + [overriding_tag]
        rm1mv1 = fs.get_model_version(name1, 1)
        assert rm1mv1.tags == {tag.key: tag.value for tag in all_tags}
        # does not affect other model versions with the same key
        rm1mv2 = fs.get_model_version(name1, 2)
        rm2mv1 = fs.get_model_version(name2, 1)
        assert rm1mv2.tags == {tag.key: tag.value for tag in initial_tags}
        assert rm2mv1.tags == {tag.key: tag.value for tag in initial_tags}

        # can not set tag on deleted (non-existed) model version
        fs.delete_model_version(name1, 2)
        with pytest.raises(
            MlflowException, match=rf"Model Version \(name={name1}, version=2\) not found"
        ) as exception_context:
            fs.set_model_version_tag(name1, 2, overriding_tag)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # test cannot set tags that are too long
        long_tag = ModelVersionTag("longTagKey", "a" * 5001)
        with pytest.raises(
            MlflowException,
            match=r"Model version value '.+' had length \d+, which exceeded length limit of 5000",
        ) as exception_context:
            fs.set_model_version_tag(name1, 1, long_tag)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # test can set tags that are somewhat long
        long_tag = ModelVersionTag("longTagKey", "a" * 4999)
        fs.set_model_version_tag(name1, 1, long_tag)
        # can not set invalid tag
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            fs.set_model_version_tag(name2, 1, ModelVersionTag(key=None, value=""))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name or version
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            fs.set_model_version_tag(None, 1, ModelVersionTag(key="key", value="value"))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        with pytest.raises(
            MlflowException, match=r"Model version must be an integer"
        ) as exception_context:
            fs.set_model_version_tag(
                name2, "I am not a version", ModelVersionTag(key="key", value="value")
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_delete_model_version_tag(self):
        fs = self.get_store()

        name1 = "DeleteModelVersionTag_TestMod"
        name2 = "DeleteModelVersionTag_TestMod 2"
        initial_tags = [
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "some other value"),
        ]
        fs.create_registered_model(name1)
        fs.create_registered_model(name2)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        fs.create_model_version(name1, "A/B", run_id_1, initial_tags)
        fs.create_model_version(name1, "A/C", run_id_2, initial_tags)
        fs.create_model_version(name2, "A/D", run_id_3, initial_tags)
        new_tag = ModelVersionTag("randomTag", "not a random value")
        fs.set_model_version_tag(name1, 1, new_tag)
        fs.delete_model_version_tag(name1, 1, "randomTag")
        rm1mv1 = fs.get_model_version(name1, 1)
        assert rm1mv1.tags == {tag.key: tag.value for tag in initial_tags}

        # testing deleting a key does not affect other model versions with the same key
        fs.delete_model_version_tag(name1, 1, "key")
        rm1mv1 = fs.get_model_version(name1, 1)
        rm1mv2 = fs.get_model_version(name1, 2)
        rm2mv1 = fs.get_model_version(name2, 1)
        assert rm1mv1.tags == {"anotherKey": "some other value"}
        assert rm1mv2.tags == {tag.key: tag.value for tag in initial_tags}
        assert rm2mv1.tags == {tag.key: tag.value for tag in initial_tags}

        # delete tag that is already deleted does nothing
        fs.delete_model_version_tag(name1, 1, "key")
        rm1mv1 = fs.get_model_version(name1, 1)
        assert rm1mv1.tags == {"anotherKey": "some other value"}

        # can not delete tag on deleted (non-existed) model version
        fs.delete_model_version(name2, 1)
        with pytest.raises(
            MlflowException, match=rf"Model Version \(name={name2}, version=1\) not found"
        ) as exception_context:
            fs.delete_model_version_tag(name2, 1, "key")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # can not delete tag with invalid key
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            fs.delete_model_version_tag(name1, 2, None)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name or version
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            fs.delete_model_version_tag(None, 2, "key")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        with pytest.raises(
            MlflowException, match=r"Model version must be an integer"
        ) as exception_context:
            fs.delete_model_version_tag(name1, "I am not a version", "key")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_pyfunc_model_registry_with_file_store(self):
        import mlflow
        from mlflow.pyfunc import PythonModel

        class MyModel(PythonModel):
            def predict(self, context, model_input):
                return 7

        fs = self.get_store()
        mlflow.set_registry_uri(path_to_local_file_uri(fs.root_directory))
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                python_model=MyModel(), artifact_path="foo", registered_model_name="model1"
            )
            mlflow.pyfunc.log_model(
                python_model=MyModel(), artifact_path="foo", registered_model_name="model2"
            )
            mlflow.pyfunc.log_model(
                python_model=MyModel(), artifact_path="foo", registered_model_name="model1"
            )

        with mlflow.start_run():
            mlflow.log_param("A", "B")

        models = fs.search_registered_models(max_results=10)
        assert len(models) == 2
        assert models[0].name == "model1"
        assert models[1].name == "model2"
        mv1 = fs.search_model_versions("name = 'model1'")
        assert len(mv1) == 2 and mv1[0].name == "model1"
        mv2 = fs.search_model_versions("name = 'model2'")
        assert len(mv2) == 1 and mv2[0].name == "model2"
