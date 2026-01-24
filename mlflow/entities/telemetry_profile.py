from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos import databricks_telemetry_profile_pb2 as pb


class ExporterType(str, Enum):
    TYPE_UNSPECIFIED = "TYPE_UNSPECIFIED"
    UNITY_CATALOG_TABLES = "UNITY_CATALOG_TABLES"

    def to_proto(self) -> int:
        return pb.Exporter.Type.Value(self.value)

    @classmethod
    def from_proto(cls, proto: int) -> "ExporterType":
        return cls(pb.Exporter.Type.Name(proto))


@dataclass
class UnityCatalogTablesConfig(_MlflowObject):
    """
    Configuration for Unity Catalog Delta table exporter.

    Args:
        uc_catalog: The UC catalog name where OTel tables will be stored.
        uc_schema: The UC schema name where OTel tables will be stored.
        uc_table_prefix: The table prefix for OTel tables.
        logs_table_name: The fully qualified name of the logs table.
        logs_schema_version: The schema version of the logs table.
        metrics_table_name: The fully qualified name of the metrics table.
        metrics_schema_version: The schema version of the metrics table.
        spans_table_name: The fully qualified name of the spans table.
        spans_schema_version: The schema version of the spans table.
    """

    uc_catalog: str | None = None
    uc_schema: str | None = None
    uc_table_prefix: str | None = None
    logs_table_name: str | None = None
    logs_schema_version: str | None = None
    metrics_table_name: str | None = None
    metrics_schema_version: str | None = None
    spans_table_name: str | None = None
    spans_schema_version: str | None = None

    def to_proto(self) -> pb.UnityCatalogTablesConfig:
        proto = pb.UnityCatalogTablesConfig()
        if self.uc_catalog is not None:
            proto.uc_catalog = self.uc_catalog
        if self.uc_schema is not None:
            proto.uc_schema = self.uc_schema
        if self.uc_table_prefix is not None:
            proto.uc_table_prefix = self.uc_table_prefix
        if self.logs_table_name is not None:
            proto.logs_table_name = self.logs_table_name
        if self.logs_schema_version is not None:
            proto.logs_schema_version = self.logs_schema_version
        if self.metrics_table_name is not None:
            proto.metrics_table_name = self.metrics_table_name
        if self.metrics_schema_version is not None:
            proto.metrics_schema_version = self.metrics_schema_version
        if self.spans_table_name is not None:
            proto.spans_table_name = self.spans_table_name
        if self.spans_schema_version is not None:
            proto.spans_schema_version = self.spans_schema_version
        return proto

    @classmethod
    def from_proto(cls, proto: pb.UnityCatalogTablesConfig) -> "UnityCatalogTablesConfig":
        return cls(
            uc_catalog=proto.uc_catalog if proto.HasField("uc_catalog") else None,
            uc_schema=proto.uc_schema if proto.HasField("uc_schema") else None,
            uc_table_prefix=proto.uc_table_prefix if proto.HasField("uc_table_prefix") else None,
            logs_table_name=proto.logs_table_name if proto.HasField("logs_table_name") else None,
            logs_schema_version=(
                proto.logs_schema_version if proto.HasField("logs_schema_version") else None
            ),
            metrics_table_name=(
                proto.metrics_table_name if proto.HasField("metrics_table_name") else None
            ),
            metrics_schema_version=(
                proto.metrics_schema_version if proto.HasField("metrics_schema_version") else None
            ),
            spans_table_name=proto.spans_table_name if proto.HasField("spans_table_name") else None,
            spans_schema_version=(
                proto.spans_schema_version if proto.HasField("spans_schema_version") else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {}
        if self.uc_catalog is not None:
            d["uc_catalog"] = self.uc_catalog
        if self.uc_schema is not None:
            d["uc_schema"] = self.uc_schema
        if self.uc_table_prefix is not None:
            d["uc_table_prefix"] = self.uc_table_prefix
        if self.logs_table_name is not None:
            d["logs_table_name"] = self.logs_table_name
        if self.logs_schema_version is not None:
            d["logs_schema_version"] = self.logs_schema_version
        if self.metrics_table_name is not None:
            d["metrics_table_name"] = self.metrics_table_name
        if self.metrics_schema_version is not None:
            d["metrics_schema_version"] = self.metrics_schema_version
        if self.spans_table_name is not None:
            d["spans_table_name"] = self.spans_table_name
        if self.spans_schema_version is not None:
            d["spans_schema_version"] = self.spans_schema_version
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnityCatalogTablesConfig":
        return cls(
            uc_catalog=d.get("uc_catalog"),
            uc_schema=d.get("uc_schema"),
            uc_table_prefix=d.get("uc_table_prefix"),
            logs_table_name=d.get("logs_table_name"),
            logs_schema_version=d.get("logs_schema_version"),
            metrics_table_name=d.get("metrics_table_name"),
            metrics_schema_version=d.get("metrics_schema_version"),
            spans_table_name=d.get("spans_table_name"),
            spans_schema_version=d.get("spans_schema_version"),
        )


@dataclass
class Exporter(_MlflowObject):
    """
    Defines an exporter destination for telemetry data.

    Args:
        type: The type of exporter.
        uc_tables: Unity Catalog tables configuration (when type is UNITY_CATALOG_TABLES).
    """

    type: ExporterType = ExporterType.TYPE_UNSPECIFIED
    uc_tables: UnityCatalogTablesConfig | None = None

    def to_proto(self) -> pb.Exporter:
        proto = pb.Exporter()
        proto.type = self.type.to_proto()
        if self.uc_tables is not None:
            proto.uc_tables.CopyFrom(self.uc_tables.to_proto())
        return proto

    @classmethod
    def from_proto(cls, proto: pb.Exporter) -> "Exporter":
        return cls(
            type=ExporterType.from_proto(proto.type),
            uc_tables=(
                UnityCatalogTablesConfig.from_proto(proto.uc_tables)
                if proto.HasField("uc_tables")
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {"type": self.type.value}
        if self.uc_tables is not None:
            d["uc_tables"] = self.uc_tables.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exporter":
        return cls(
            type=ExporterType(d["type"]),
            uc_tables=(
                UnityCatalogTablesConfig.from_dict(d["uc_tables"]) if "uc_tables" in d else None
            ),
        )


@dataclass
class TelemetryProfile(_MlflowObject):
    """
    Telemetry profile defines OpenTelemetry collector configuration.

    Args:
        profile_id: The profile ID (user-provided or auto-generated UUID).
        profile_name: The name of the profile.
        created_at: Unix timestamp in milliseconds when the profile was created.
        created_by: User email of the creator.
        updated_at: Unix timestamp in milliseconds when the profile was last updated.
        updated_by: User email of the last updater.
        exporters: List of exporter configurations.
    """

    profile_id: str | None = None
    profile_name: str | None = None
    created_at: int | None = None
    created_by: str | None = None
    updated_at: int | None = None
    updated_by: str | None = None
    exporters: list[Exporter] = field(default_factory=list)

    def to_proto(self) -> pb.TelemetryProfile:
        proto = pb.TelemetryProfile()
        if self.profile_id is not None:
            proto.profile_id = self.profile_id
        if self.profile_name is not None:
            proto.profile_name = self.profile_name
        if self.created_at is not None:
            proto.created_at = self.created_at
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.updated_at is not None:
            proto.updated_at = self.updated_at
        if self.updated_by is not None:
            proto.updated_by = self.updated_by
        for exporter in self.exporters:
            proto.exporters.append(exporter.to_proto())
        return proto

    @classmethod
    def from_proto(cls, proto: pb.TelemetryProfile) -> "TelemetryProfile":
        return cls(
            profile_id=proto.profile_id if proto.HasField("profile_id") else None,
            profile_name=proto.profile_name if proto.HasField("profile_name") else None,
            created_at=proto.created_at if proto.HasField("created_at") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            updated_at=proto.updated_at if proto.HasField("updated_at") else None,
            updated_by=proto.updated_by if proto.HasField("updated_by") else None,
            exporters=[Exporter.from_proto(e) for e in proto.exporters],
        )

    def to_dict(self) -> dict[str, Any]:
        d = {}
        if self.profile_id is not None:
            d["profile_id"] = self.profile_id
        if self.profile_name is not None:
            d["profile_name"] = self.profile_name
        if self.created_at is not None:
            d["created_at"] = self.created_at
        if self.created_by is not None:
            d["created_by"] = self.created_by
        if self.updated_at is not None:
            d["updated_at"] = self.updated_at
        if self.updated_by is not None:
            d["updated_by"] = self.updated_by
        if self.exporters:
            d["exporters"] = [e.to_dict() for e in self.exporters]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TelemetryProfile":
        return cls(
            profile_id=d.get("profile_id"),
            profile_name=d.get("profile_name"),
            created_at=d.get("created_at"),
            created_by=d.get("created_by"),
            updated_at=d.get("updated_at"),
            updated_by=d.get("updated_by"),
            exporters=[Exporter.from_dict(e) for e in d.get("exporters", [])],
        )

    def get_uc_tables_config(self) -> UnityCatalogTablesConfig | None:
        """
        Get the first Unity Catalog tables configuration from exporters.

        Returns:
            The UnityCatalogTablesConfig if found, None otherwise.
        """
        for exporter in self.exporters:
            if exporter.type == ExporterType.UNITY_CATALOG_TABLES and exporter.uc_tables:
                return exporter.uc_tables
        return None
