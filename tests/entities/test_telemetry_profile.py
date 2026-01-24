import pytest

from mlflow.entities.telemetry_profile import (
    Exporter,
    ExporterType,
    TelemetryProfile,
    UnityCatalogTablesConfig,
)
from mlflow.protos import databricks_telemetry_profile_pb2 as pb


def test_unity_catalog_tables_config():
    config = UnityCatalogTablesConfig(
        uc_catalog="test_catalog",
        uc_schema="test_schema",
        uc_table_prefix="prefix_",
        spans_table_name="test_catalog.test_schema.prefix_otel_spans",
    )
    assert config.uc_catalog == "test_catalog"
    assert config.uc_schema == "test_schema"
    assert config.uc_table_prefix == "prefix_"
    assert config.spans_table_name == "test_catalog.test_schema.prefix_otel_spans"


def test_unity_catalog_tables_config_to_dict():
    config = UnityCatalogTablesConfig(
        uc_catalog="cat",
        uc_schema="sch",
        uc_table_prefix="pre_",
    )
    d = config.to_dict()
    assert d == {
        "uc_catalog": "cat",
        "uc_schema": "sch",
        "uc_table_prefix": "pre_",
    }


def test_unity_catalog_tables_config_from_dict():
    d = {
        "uc_catalog": "cat",
        "uc_schema": "sch",
        "uc_table_prefix": "pre_",
        "spans_table_name": "cat.sch.pre_otel_spans",
    }
    config = UnityCatalogTablesConfig.from_dict(d)
    assert config.uc_catalog == "cat"
    assert config.uc_schema == "sch"
    assert config.uc_table_prefix == "pre_"
    assert config.spans_table_name == "cat.sch.pre_otel_spans"


def test_unity_catalog_tables_config_to_from_proto():
    config = UnityCatalogTablesConfig(
        uc_catalog="cat",
        uc_schema="sch",
        uc_table_prefix="pre_",
        spans_table_name="cat.sch.pre_otel_spans",
        spans_schema_version="1.0",
    )
    proto = config.to_proto()
    assert proto.uc_catalog == "cat"
    assert proto.uc_schema == "sch"
    assert proto.uc_table_prefix == "pre_"
    assert proto.spans_table_name == "cat.sch.pre_otel_spans"

    config2 = UnityCatalogTablesConfig.from_proto(proto)
    assert config2.uc_catalog == config.uc_catalog
    assert config2.uc_schema == config.uc_schema
    assert config2.uc_table_prefix == config.uc_table_prefix
    assert config2.spans_table_name == config.spans_table_name
    assert config2.spans_schema_version == config.spans_schema_version


def test_exporter_type_enum():
    assert ExporterType.TYPE_UNSPECIFIED.value == "TYPE_UNSPECIFIED"
    assert ExporterType.UNITY_CATALOG_TABLES.value == "UNITY_CATALOG_TABLES"


def test_exporter_type_to_from_proto():
    assert ExporterType.UNITY_CATALOG_TABLES.to_proto() == pb.Exporter.Type.UNITY_CATALOG_TABLES
    assert ExporterType.from_proto(pb.Exporter.Type.UNITY_CATALOG_TABLES) == (
        ExporterType.UNITY_CATALOG_TABLES
    )


def test_exporter():
    config = UnityCatalogTablesConfig(uc_catalog="cat", uc_schema="sch")
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    assert exporter.type == ExporterType.UNITY_CATALOG_TABLES
    assert exporter.uc_tables.uc_catalog == "cat"


def test_exporter_to_dict():
    config = UnityCatalogTablesConfig(uc_catalog="cat", uc_schema="sch")
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    d = exporter.to_dict()
    assert d == {
        "type": "UNITY_CATALOG_TABLES",
        "uc_tables": {"uc_catalog": "cat", "uc_schema": "sch"},
    }


def test_exporter_from_dict():
    d = {
        "type": "UNITY_CATALOG_TABLES",
        "uc_tables": {"uc_catalog": "cat", "uc_schema": "sch"},
    }
    exporter = Exporter.from_dict(d)
    assert exporter.type == ExporterType.UNITY_CATALOG_TABLES
    assert exporter.uc_tables.uc_catalog == "cat"
    assert exporter.uc_tables.uc_schema == "sch"


def test_exporter_to_from_proto():
    config = UnityCatalogTablesConfig(uc_catalog="cat", uc_schema="sch", uc_table_prefix="pre_")
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)

    proto = exporter.to_proto()
    assert proto.type == pb.Exporter.Type.UNITY_CATALOG_TABLES
    assert proto.uc_tables.uc_catalog == "cat"

    exporter2 = Exporter.from_proto(proto)
    assert exporter2.type == exporter.type
    assert exporter2.uc_tables.uc_catalog == exporter.uc_tables.uc_catalog


def test_telemetry_profile():
    config = UnityCatalogTablesConfig(
        uc_catalog="cat",
        uc_schema="sch",
        uc_table_prefix="pre_",
        spans_table_name="cat.sch.pre_otel_spans",
    )
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    profile = TelemetryProfile(
        profile_id="profile-123",
        profile_name="Test Profile",
        created_at=1234567890,
        created_by="user@example.com",
        exporters=[exporter],
    )
    assert profile.profile_id == "profile-123"
    assert profile.profile_name == "Test Profile"
    assert profile.created_at == 1234567890
    assert profile.created_by == "user@example.com"
    assert len(profile.exporters) == 1
    assert profile.exporters[0].type == ExporterType.UNITY_CATALOG_TABLES


def test_telemetry_profile_to_dict():
    exporter = Exporter(
        type=ExporterType.UNITY_CATALOG_TABLES,
        uc_tables=UnityCatalogTablesConfig(uc_catalog="cat", uc_schema="sch"),
    )
    profile = TelemetryProfile(
        profile_id="p1",
        profile_name="Profile 1",
        exporters=[exporter],
    )
    d = profile.to_dict()
    assert d == {
        "profile_id": "p1",
        "profile_name": "Profile 1",
        "exporters": [
            {
                "type": "UNITY_CATALOG_TABLES",
                "uc_tables": {"uc_catalog": "cat", "uc_schema": "sch"},
            }
        ],
    }


def test_telemetry_profile_from_dict():
    d = {
        "profile_id": "p1",
        "profile_name": "Profile 1",
        "created_at": 123,
        "exporters": [
            {
                "type": "UNITY_CATALOG_TABLES",
                "uc_tables": {"uc_catalog": "cat", "uc_schema": "sch"},
            }
        ],
    }
    profile = TelemetryProfile.from_dict(d)
    assert profile.profile_id == "p1"
    assert profile.profile_name == "Profile 1"
    assert profile.created_at == 123
    assert len(profile.exporters) == 1
    assert profile.exporters[0].type == ExporterType.UNITY_CATALOG_TABLES
    assert profile.exporters[0].uc_tables.uc_catalog == "cat"


def test_telemetry_profile_to_from_proto():
    config = UnityCatalogTablesConfig(
        uc_catalog="cat",
        uc_schema="sch",
        uc_table_prefix="pre_",
    )
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    profile = TelemetryProfile(
        profile_id="p1",
        profile_name="Profile",
        created_at=1000,
        created_by="user@test.com",
        updated_at=2000,
        updated_by="user2@test.com",
        exporters=[exporter],
    )

    proto = profile.to_proto()
    assert proto.profile_id == "p1"
    assert proto.profile_name == "Profile"
    assert proto.created_at == 1000
    assert len(proto.exporters) == 1

    profile2 = TelemetryProfile.from_proto(proto)
    assert profile2.profile_id == profile.profile_id
    assert profile2.profile_name == profile.profile_name
    assert profile2.created_at == profile.created_at
    assert profile2.created_by == profile.created_by
    assert profile2.updated_at == profile.updated_at
    assert profile2.updated_by == profile.updated_by
    assert len(profile2.exporters) == 1
    assert profile2.exporters[0].type == ExporterType.UNITY_CATALOG_TABLES


def test_telemetry_profile_get_uc_tables_config():
    config = UnityCatalogTablesConfig(
        uc_catalog="cat",
        uc_schema="sch",
        uc_table_prefix="pre_",
        spans_table_name="cat.sch.pre_otel_spans",
    )
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    profile = TelemetryProfile(exporters=[exporter])

    uc_config = profile.get_uc_tables_config()
    assert uc_config is not None
    assert uc_config.uc_catalog == "cat"
    assert uc_config.uc_schema == "sch"
    assert uc_config.uc_table_prefix == "pre_"
    assert uc_config.spans_table_name == "cat.sch.pre_otel_spans"


def test_telemetry_profile_get_uc_tables_config_empty():
    profile = TelemetryProfile(exporters=[])
    assert profile.get_uc_tables_config() is None


def test_telemetry_profile_get_uc_tables_config_wrong_type():
    exporter = Exporter(type=ExporterType.TYPE_UNSPECIFIED)
    profile = TelemetryProfile(exporters=[exporter])
    assert profile.get_uc_tables_config() is None
