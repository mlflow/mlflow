# Trace View Ranges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor TraceView from a single SpanFilter model to a list of SpanRange entries, each with from/to selectors, label, description, and JSONPath extraction.

**Architecture:** Replace `SpanFilter` with `SpanSelector` (adds `span_id` field). Introduce `SpanRange` dataclass linking two selectors with metadata. `TraceView.ranges` replaces the old `span_filter`/`input_path`/`output_path`/`description` fields. A new `trace_view_ranges` DB table stores ranges with cascade delete from `trace_views`. The DFS resolution algorithm walks the span tree to resolve ranges at runtime.

**Tech Stack:** Python (dataclasses, SQLAlchemy, alembic), `jsonpath-ng` (Python), React/TypeScript, `jsonpath-plus` (JS)

**Spec:** `docs/superpowers/specs/2026-04-01-trace-view-ranges-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| (none — all changes modify existing files) | |

### Modified Files

| File | Change |
|------|--------|
| `mlflow/entities/trace_view.py` | Rename `SpanFilter` → `SpanSelector` (add `span_id`), add `SpanRange` dataclass, update `TraceView` to carry `ranges: list[SpanRange]` instead of single filter |
| `mlflow/store/tracking/dbmodels/models.py:1125-1231` | Update `SqlTraceView` (drop filter/path/description columns), add `SqlTraceViewRange` model |
| `mlflow/store/db_migrations/versions/eb885a9619f6_add_trace_views_table.py` | Rewrite migration: simplified `trace_views` + new `trace_view_ranges` table |
| `mlflow/store/tracking/abstract_store.py:605-628` | Update `update_trace_view` signature (remove old params, add `ranges`) |
| `mlflow/store/tracking/sqlalchemy_store.py:4210-4341` | Update CRUD to handle ranges (create/fetch ranges alongside views, replace-all on update) |
| `mlflow/store/tracking/rest_store.py:991-1066` | Update request/response payloads for ranges |
| `mlflow/store/tracking/sqlalchemy_workspace_store.py:496-695` | Update workspace-aware overrides to match new signatures and handle ranges |
| `mlflow/server/handlers.py:4187-4320` | Update create/update handlers for ranges payload |
| `mlflow/tracking/client.py:6713-6771` | Update `create_trace_view`/`update_trace_view` signatures for ranges |
| `mlflow/tracking/_tracking_service/client.py:1132-1166` | Update service-layer signatures for ranges |
| `mlflow/tracing/utils/view_utils.py` | Replace `apply_view`/`find_first_matching_span` with DFS resolution: `resolve_range`, `resolve_view` |
| `mlflow/entities/trace.py:298-366` | Update `create_view`, `summarize`, `analyze` for new model |
| `tests/entities/test_trace_view.py` | Rewrite tests for SpanSelector, SpanRange, TraceView |
| `tests/store/tracking/test_trace_views.py` | Rewrite store CRUD tests for ranges |
| `tests/tracing/utils/test_view_utils.py` | Rewrite for DFS resolution |

---

## Task 1: Entity Classes — SpanSelector, SpanRange, TraceView

**Files:**
- Modify: `mlflow/entities/trace_view.py`
- Test: `tests/entities/test_trace_view.py`

- [ ] **Step 1: Write failing tests for SpanSelector**

Replace the entire contents of `tests/entities/test_trace_view.py`:

```python
from __future__ import annotations

import json

import pytest

from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.exceptions import MlflowException


class TestSpanSelector:
    def test_creation_defaults(self):
        s = SpanSelector()
        assert s.span_name is None
        assert s.span_type is None
        assert s.span_id is None
        assert s.attribute_key is None
        assert s.attribute_value is None

    def test_creation_with_values(self):
        s = SpanSelector(
            span_name="my_span",
            span_type="LLM",
            span_id="span-1",
            attribute_key="model",
            attribute_value="gpt-4",
        )
        assert s.span_name == "my_span"
        assert s.span_type == "LLM"
        assert s.span_id == "span-1"

    def test_to_dict(self):
        s = SpanSelector(span_name="my_span", span_type="LLM")
        d = s.to_dict()
        assert d == {
            "span_name": "my_span",
            "span_type": "LLM",
            "span_id": None,
            "attribute_key": None,
            "attribute_value": None,
        }

    def test_from_dict(self):
        d = {"span_name": "x", "span_type": "CHAIN", "span_id": "s1"}
        s = SpanSelector.from_dict(d)
        assert s.span_name == "x"
        assert s.span_type == "CHAIN"
        assert s.span_id == "s1"

    def test_from_dict_partial(self):
        s = SpanSelector.from_dict({"span_name": "only_name"})
        assert s.span_name == "only_name"
        assert s.span_type is None
        assert s.span_id is None

    def test_json_round_trip(self):
        s = SpanSelector(span_name="test", span_id="s1", attribute_key="key")
        json_str = s.to_json()
        restored = SpanSelector.from_json(json_str)
        assert restored == s

    def test_to_json_is_valid_json(self):
        s = SpanSelector(span_name="test")
        parsed = json.loads(s.to_json())
        assert isinstance(parsed, dict)


class TestSpanRange:
    def test_creation_minimal(self):
        r = SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            label="LLM Call",
            description="An LLM call",
        )
        assert r.from_selector.span_type == "LLM"
        assert r.to_selector is None
        assert r.label == "LLM Call"
        assert r.input_path is None
        assert r.output_path is None
        assert r.position == 0

    def test_creation_full(self):
        r = SpanRange(
            from_selector=SpanSelector(span_id="s1"),
            to_selector=SpanSelector(span_name="search"),
            label="Template Lookup",
            description="Searched for template",
            input_path="$.reasoning",
            output_path="$.results",
            position=1,
            range_id="r-abc",
        )
        assert r.to_selector.span_name == "search"
        assert r.input_path == "$.reasoning"
        assert r.range_id == "r-abc"
        assert r.position == 1

    def test_to_dict_round_trip(self):
        r = SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            to_selector=SpanSelector(span_name="tool"),
            label="Step 1",
            description="First step",
            input_path="$.input",
            output_path="$.output",
            position=0,
            range_id="r-1",
        )
        d = r.to_dict()
        restored = SpanRange.from_dict(d)
        assert restored.from_selector == r.from_selector
        assert restored.to_selector == r.to_selector
        assert restored.label == r.label
        assert restored.description == r.description
        assert restored.input_path == r.input_path
        assert restored.output_path == r.output_path
        assert restored.position == r.position
        assert restored.range_id == r.range_id

    def test_to_dict_without_to_selector(self):
        r = SpanRange(
            from_selector=SpanSelector(span_id="s1"),
            label="Solo",
            description="Single span",
        )
        d = r.to_dict()
        assert d["to_selector"] is None
        restored = SpanRange.from_dict(d)
        assert restored.to_selector is None


class TestTraceView:
    def test_trace_scoped_creation(self):
        tv = TraceView(name="my_view", trace_id="tr-123")
        assert tv.name == "my_view"
        assert tv.trace_id == "tr-123"
        assert tv.experiment_id is None
        assert tv.ranges == []

    def test_experiment_scoped_creation(self):
        tv = TraceView(name="exp_view", experiment_id="exp-456")
        assert tv.experiment_id == "exp-456"
        assert tv.trace_id is None

    def test_scope_property(self):
        assert TraceView(name="v", trace_id="t1").scope == "trace"
        assert TraceView(name="v", experiment_id="e1").scope == "experiment"

    def test_validate_scope_both_set_raises(self):
        tv = TraceView(name="v", trace_id="t1", experiment_id="e1")
        with pytest.raises(MlflowException, match="Exactly one of"):
            tv.validate_scope()

    def test_validate_scope_neither_set_raises(self):
        tv = TraceView(name="v")
        with pytest.raises(MlflowException, match="Exactly one of"):
            tv.validate_scope()

    def test_validate_scope_valid(self):
        TraceView(name="v", trace_id="t1").validate_scope()
        TraceView(name="v", experiment_id="e1").validate_scope()

    def test_to_dict_round_trip_with_ranges(self):
        tv = TraceView(
            name="view1",
            trace_id="tr-1",
            ranges=[
                SpanRange(
                    from_selector=SpanSelector(span_name="root"),
                    label="Summary",
                    description="A summary",
                    position=0,
                    range_id="r-1",
                ),
                SpanRange(
                    from_selector=SpanSelector(span_type="LLM"),
                    to_selector=SpanSelector(span_type="TOOL"),
                    label="Step 1",
                    description="First step",
                    input_path="$.reasoning",
                    position=1,
                    range_id="r-2",
                ),
            ],
            created_by="assistant",
            view_id="vid-1",
            create_time_ms=1000,
            last_update_time_ms=2000,
        )
        d = tv.to_dict()
        restored = TraceView.from_dict(d)
        assert restored.name == tv.name
        assert restored.trace_id == tv.trace_id
        assert len(restored.ranges) == 2
        assert restored.ranges[0].label == "Summary"
        assert restored.ranges[1].to_selector.span_type == "TOOL"
        assert restored.ranges[1].input_path == "$.reasoning"
        assert restored.created_by == tv.created_by
        assert restored.view_id == tv.view_id

    def test_to_dict_round_trip_no_ranges(self):
        tv = TraceView(name="empty", trace_id="t")
        d = tv.to_dict()
        restored = TraceView.from_dict(d)
        assert restored.ranges == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/entities/test_trace_view.py -v`
Expected: FAIL — `SpanSelector` and `SpanRange` do not exist yet.

- [ ] **Step 3: Implement SpanSelector, SpanRange, and updated TraceView**

Replace the entire contents of `mlflow/entities/trace_view.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class SpanSelector:
    span_name: str | None = None
    span_type: str | None = None
    span_id: str | None = None
    attribute_key: str | None = None
    attribute_value: str | None = None

    def to_dict(self) -> dict:
        return {
            "span_name": self.span_name,
            "span_type": self.span_type,
            "span_id": self.span_id,
            "attribute_key": self.attribute_key,
            "attribute_value": self.attribute_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpanSelector:
        return cls(
            span_name=d.get("span_name"),
            span_type=d.get("span_type"),
            span_id=d.get("span_id"),
            attribute_key=d.get("attribute_key"),
            attribute_value=d.get("attribute_value"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> SpanSelector:
        return cls.from_dict(json.loads(s))


@dataclass
class SpanRange:
    from_selector: SpanSelector
    to_selector: SpanSelector | None = None
    label: str = ""
    description: str = ""
    input_path: str | None = None
    output_path: str | None = None
    position: int = 0
    range_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "from_selector": self.from_selector.to_dict(),
            "to_selector": self.to_selector.to_dict() if self.to_selector else None,
            "label": self.label,
            "description": self.description,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "position": self.position,
            "range_id": self.range_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpanRange:
        to_sel = d.get("to_selector")
        return cls(
            from_selector=SpanSelector.from_dict(d["from_selector"]),
            to_selector=SpanSelector.from_dict(to_sel) if to_sel else None,
            label=d.get("label", ""),
            description=d.get("description", ""),
            input_path=d.get("input_path"),
            output_path=d.get("output_path"),
            position=d.get("position", 0),
            range_id=d.get("range_id"),
        )


@dataclass
class TraceView:
    name: str
    trace_id: str | None = None
    experiment_id: str | None = None
    ranges: list[SpanRange] = field(default_factory=list)
    created_by: str | None = None
    view_id: str | None = None
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None

    def validate_scope(self) -> None:
        has_trace = self.trace_id is not None
        has_experiment = self.experiment_id is not None
        if has_trace == has_experiment:
            raise MlflowException(
                "Exactly one of trace_id or experiment_id must be set.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @property
    def scope(self) -> str:
        if self.trace_id is not None:
            return "trace"
        return "experiment"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "experiment_id": self.experiment_id,
            "ranges": [r.to_dict() for r in self.ranges],
            "created_by": self.created_by,
            "view_id": self.view_id,
            "create_time_ms": self.create_time_ms,
            "last_update_time_ms": self.last_update_time_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceView:
        ranges_data = d.get("ranges", [])
        return cls(
            name=d["name"],
            trace_id=d.get("trace_id"),
            experiment_id=d.get("experiment_id"),
            ranges=[SpanRange.from_dict(r) for r in ranges_data],
            created_by=d.get("created_by"),
            view_id=d.get("view_id"),
            create_time_ms=d.get("create_time_ms"),
            last_update_time_ms=d.get("last_update_time_ms"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/entities/test_trace_view.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mlflow/entities/trace_view.py tests/entities/test_trace_view.py
git commit -s -m "refactor: replace SpanFilter with SpanSelector, add SpanRange to TraceView

Renames SpanFilter to SpanSelector (adds span_id field), introduces
SpanRange dataclass, and updates TraceView to carry a list of ranges
instead of a single filter.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Database Models — SqlTraceView + SqlTraceViewRange

**Files:**
- Modify: `mlflow/store/tracking/dbmodels/models.py:1125-1231`
- Modify: `mlflow/store/db_migrations/versions/eb885a9619f6_add_trace_views_table.py`

- [ ] **Step 1: Update SqlTraceView and add SqlTraceViewRange**

In `mlflow/store/tracking/dbmodels/models.py`, replace the `SqlTraceView` class (lines 1125-1231) with:

```python
class SqlTraceView(Base):
    __tablename__ = "trace_views"

    view_id = Column(String(50), nullable=False)
    name = Column(String(256), nullable=False)
    trace_id = Column(
        String(50), ForeignKey("trace_info.request_id", ondelete="CASCADE"), nullable=True
    )
    experiment_id = Column(
        Integer, ForeignKey("experiments.experiment_id"), nullable=True
    )
    created_by = Column(String(256), nullable=True)
    created_timestamp = Column(BigInteger, nullable=False)
    last_updated_timestamp = Column(BigInteger, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("view_id", name="trace_views_pk"),
        CheckConstraint(
            "(trace_id IS NOT NULL AND experiment_id IS NULL) OR "
            "(trace_id IS NULL AND experiment_id IS NOT NULL)",
            name="ck_trace_views_scope",
        ),
    )

    ranges = relationship(
        "SqlTraceViewRange",
        back_populates="view",
        cascade="all, delete-orphan",
        order_by="SqlTraceViewRange.position",
        lazy="joined",
    )

    def to_mlflow_entity(self):
        from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView

        mlflow_ranges = []
        for sql_range in self.ranges:
            from_sel = SpanSelector.from_json(sql_range.from_selector)
            to_sel = SpanSelector.from_json(sql_range.to_selector) if sql_range.to_selector else None
            mlflow_ranges.append(
                SpanRange(
                    from_selector=from_sel,
                    to_selector=to_sel,
                    label=sql_range.label,
                    description=sql_range.description,
                    input_path=sql_range.input_path,
                    output_path=sql_range.output_path,
                    position=sql_range.position,
                    range_id=sql_range.range_id,
                )
            )

        return TraceView(
            view_id=self.view_id,
            name=self.name,
            trace_id=self.trace_id,
            experiment_id=str(self.experiment_id) if self.experiment_id is not None else None,
            ranges=mlflow_ranges,
            created_by=self.created_by,
            create_time_ms=self.created_timestamp,
            last_update_time_ms=self.last_updated_timestamp,
        )

    @classmethod
    def from_mlflow_entity(cls, view):
        current_timestamp = int(time.time() * 1000)
        view_id = view.view_id or f"tv-{uuid.uuid4().hex[:12]}"
        experiment_id = int(view.experiment_id) if view.experiment_id is not None else None

        sql_view = cls(
            view_id=view_id,
            name=view.name,
            trace_id=view.trace_id,
            experiment_id=experiment_id,
            created_by=view.created_by,
            created_timestamp=view.create_time_ms or current_timestamp,
            last_updated_timestamp=view.last_update_time_ms or current_timestamp,
        )

        for i, r in enumerate(view.ranges):
            sql_view.ranges.append(
                SqlTraceViewRange(
                    range_id=r.range_id or f"tvr-{uuid.uuid4().hex[:12]}",
                    view_id=view_id,
                    position=i,
                    label=r.label,
                    description=r.description,
                    from_selector=r.from_selector.to_json(),
                    to_selector=r.to_selector.to_json() if r.to_selector else None,
                    input_path=r.input_path,
                    output_path=r.output_path,
                )
            )

        return sql_view

    def __repr__(self):
        return f"<SqlTraceView({self.view_id}, {self.name})>"


class SqlTraceViewRange(Base):
    __tablename__ = "trace_view_ranges"

    range_id = Column(String(50), nullable=False)
    view_id = Column(
        String(50), ForeignKey("trace_views.view_id", ondelete="CASCADE"), nullable=False
    )
    position = Column(Integer, nullable=False)
    label = Column(String(256), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    from_selector = Column(Text, nullable=False)
    to_selector = Column(Text, nullable=True)
    input_path = Column(Text, nullable=True)
    output_path = Column(Text, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("range_id", name="trace_view_ranges_pk"),
        UniqueConstraint("view_id", "position", name="uq_trace_view_ranges_view_position"),
    )

    view = relationship("SqlTraceView", back_populates="ranges")

    def __repr__(self):
        return f"<SqlTraceViewRange({self.range_id}, pos={self.position})>"
```

Note: Add the following imports at the top of the file if not already present: `relationship`, `UniqueConstraint`. Check existing imports — `relationship` is likely already imported from `sqlalchemy.orm`, and `UniqueConstraint` from `sqlalchemy`.

- [ ] **Step 2: Rewrite the alembic migration**

Replace the entire contents of `mlflow/store/db_migrations/versions/eb885a9619f6_add_trace_views_table.py`:

```python
"""add trace_views and trace_view_ranges tables

Create Date: 2026-03-26 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "eb885a9619f6"
down_revision = "c3d6457b6d8a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "trace_views",
        sa.Column("view_id", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("trace_id", sa.String(length=50), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("created_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_trace_views_trace_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_trace_views_experiment_id",
        ),
        sa.PrimaryKeyConstraint("view_id", name="trace_views_pk"),
        sa.CheckConstraint(
            "(trace_id IS NOT NULL AND experiment_id IS NULL) OR "
            "(trace_id IS NULL AND experiment_id IS NOT NULL)",
            name="ck_trace_views_scope",
        ),
    )

    with op.batch_alter_table("trace_views", schema=None) as batch_op:
        batch_op.create_index(
            "index_trace_views_trace_id_created_timestamp",
            ["trace_id", "created_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            "index_trace_views_experiment_id_created_timestamp",
            ["experiment_id", "created_timestamp"],
            unique=False,
        )

    op.create_table(
        "trace_view_ranges",
        sa.Column("range_id", sa.String(length=50), nullable=False),
        sa.Column("view_id", sa.String(length=50), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("from_selector", sa.Text(), nullable=False),
        sa.Column("to_selector", sa.Text(), nullable=True),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["view_id"],
            ["trace_views.view_id"],
            name="fk_trace_view_ranges_view_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("range_id", name="trace_view_ranges_pk"),
        sa.UniqueConstraint("view_id", "position", name="uq_trace_view_ranges_view_position"),
    )

    with op.batch_alter_table("trace_view_ranges", schema=None) as batch_op:
        batch_op.create_index(
            "index_trace_view_ranges_view_id_position",
            ["view_id", "position"],
            unique=False,
        )


def downgrade():
    op.drop_table("trace_view_ranges")
    op.drop_table("trace_views")
```

- [ ] **Step 3: Commit**

```bash
git add mlflow/store/tracking/dbmodels/models.py mlflow/store/db_migrations/versions/eb885a9619f6_add_trace_views_table.py
git commit -s -m "refactor: update DB models for SpanRange-based trace views

Simplifies SqlTraceView (drops span_filter, input_path, output_path,
description columns), adds SqlTraceViewRange with cascade delete.
Rewrites alembic migration for both tables.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Store Layer — Abstract Store + SQLAlchemy Store

**Files:**
- Modify: `mlflow/store/tracking/abstract_store.py:605-628`
- Modify: `mlflow/store/tracking/sqlalchemy_store.py:4210-4341`
- Test: `tests/store/tracking/test_trace_views.py`

- [ ] **Step 1: Write failing store tests**

Replace the entire contents of `tests/store/tracking/test_trace_views.py`:

```python
import uuid
from pathlib import Path

import pytest

from mlflow.entities import TraceInfo, trace_location
from mlflow.entities.trace_info import TraceState
from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis

DB_URI = "sqlite:///"


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    db_uri = f"{DB_URI}{tmp_path / 'test.db'}"
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store):
    return store.create_experiment("test-experiment")


@pytest.fixture
def trace_id(store, experiment_id):
    timestamp_ms = get_current_time_millis()
    tid = f"tr-{uuid.uuid4()}"
    trace_info = store.start_trace(
        TraceInfo(
            trace_id=tid,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=timestamp_ms,
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"tr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )
    return trace_info.trace_id


def _make_ranges():
    return [
        SpanRange(
            from_selector=SpanSelector(span_name="AgentExecutor"),
            label="Summary",
            description="Agent ran and succeeded.",
        ),
        SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            to_selector=SpanSelector(span_type="TOOL"),
            label="Step 1",
            description="LLM called a tool.",
            input_path="$.reasoning",
            output_path="$.result",
        ),
    ]


def test_create_and_get_with_ranges(store, trace_id):
    view = TraceView(name="my-view", trace_id=trace_id, ranges=_make_ranges())
    created = store.create_trace_view(view)

    assert created.view_id is not None
    assert created.name == "my-view"
    assert len(created.ranges) == 2
    assert created.ranges[0].label == "Summary"
    assert created.ranges[0].range_id is not None
    assert created.ranges[1].label == "Step 1"
    assert created.ranges[1].to_selector.span_type == "TOOL"
    assert created.ranges[1].input_path == "$.reasoning"
    assert created.ranges[1].position == 1

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert len(fetched.ranges) == 2
    assert fetched.ranges[0].label == "Summary"
    assert fetched.ranges[1].output_path == "$.result"


def test_create_without_ranges(store, trace_id):
    view = TraceView(name="empty", trace_id=trace_id)
    created = store.create_trace_view(view)
    assert created.ranges == []

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert fetched.ranges == []


def test_create_experiment_scoped_with_ranges(store, experiment_id):
    view = TraceView(
        name="exp-view",
        experiment_id=experiment_id,
        ranges=[SpanRange(from_selector=SpanSelector(span_type="TOOL"), label="Tools")],
    )
    created = store.create_trace_view(view)
    assert created.experiment_id == experiment_id
    assert len(created.ranges) == 1


def test_list_views_returns_ranges(store, experiment_id, trace_id):
    store.create_trace_view(
        TraceView(name="v1", trace_id=trace_id, ranges=_make_ranges())
    )
    store.create_trace_view(
        TraceView(name="v2", experiment_id=experiment_id)
    )

    views = store.list_trace_views(trace_id=trace_id)
    names = {v.name for v in views}
    assert "v1" in names
    assert "v2" in names
    v1 = next(v for v in views if v.name == "v1")
    assert len(v1.ranges) == 2


def test_update_replaces_ranges(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="original", trace_id=trace_id, ranges=_make_ranges())
    )
    assert len(created.ranges) == 2

    new_ranges = [
        SpanRange(
            from_selector=SpanSelector(span_name="root"),
            label="New Summary",
            description="Updated summary.",
        ),
    ]
    updated = store.update_trace_view(
        view_id=created.view_id,
        ranges=new_ranges,
    )
    assert len(updated.ranges) == 1
    assert updated.ranges[0].label == "New Summary"

    fetched = store.get_trace_view(trace_id=trace_id, view_id=created.view_id)
    assert len(fetched.ranges) == 1


def test_update_name_only_preserves_ranges(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="original", trace_id=trace_id, ranges=_make_ranges())
    )

    updated = store.update_trace_view(
        view_id=created.view_id,
        name="renamed",
    )
    assert updated.name == "renamed"
    assert len(updated.ranges) == 2


def test_delete_cascades_ranges(store, trace_id):
    created = store.create_trace_view(
        TraceView(name="to-delete", trace_id=trace_id, ranges=_make_ranges())
    )
    store.delete_trace_view(view_id=created.view_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_trace_view(trace_id=trace_id, view_id=created.view_id)


def test_create_invalid_scope_raises(store):
    view = TraceView(name="bad", trace_id="t1", experiment_id="e1")
    with pytest.raises(MlflowException, match="Exactly one of"):
        store.create_trace_view(view)


def test_get_nonexistent_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_trace_view(trace_id=None, view_id="tv-nonexistent")


def test_delete_nonexistent_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_trace_view(view_id="tv-nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/store/tracking/test_trace_views.py -v`
Expected: FAIL — store methods still use old signatures.

- [ ] **Step 3: Update abstract_store.py**

In `mlflow/store/tracking/abstract_store.py`, replace the view methods (lines 605-628) with:

```python
    @abstractmethod
    def create_trace_view(self, view):
        pass

    @abstractmethod
    def get_trace_view(self, trace_id, view_id):
        pass

    @abstractmethod
    def list_trace_views(self, trace_id=None, experiment_id=None):
        pass

    @abstractmethod
    def update_trace_view(self, view_id="", name=None, ranges=None):
        pass

    @abstractmethod
    def delete_trace_view(self, trace_id=None, experiment_id=None, view_id=""):
        pass
```

- [ ] **Step 4: Update sqlalchemy_store.py**

In `mlflow/store/tracking/sqlalchemy_store.py`, replace the view CRUD methods (lines 4210-4341) with:

```python
    def create_trace_view(self, view):
        with self.ManagedSessionMaker() as session:
            view.validate_scope()
            if view.trace_id is not None:
                self._validate_trace_accessible(session, view.trace_id)
                trace_exists = (
                    session.query(SqlTraceInfo)
                    .filter(SqlTraceInfo.request_id == view.trace_id)
                    .first()
                    is not None
                )
                if not trace_exists:
                    raise MlflowException(
                        f"Trace with ID '{view.trace_id}' not found in the tracking store. "
                        f"Trace-scoped views require the trace to exist in the database. "
                        f"If this trace was fetched from a remote backend, try creating an "
                        f"experiment-scoped view with experiment_id instead.",
                        RESOURCE_DOES_NOT_EXIST,
                    )
            sql_view = SqlTraceView.from_mlflow_entity(view)
            session.add(sql_view)
            session.flush()
            return sql_view.to_mlflow_entity()

    def get_trace_view(self, trace_id, view_id):
        with self.ManagedSessionMaker() as session:
            sql_view = (
                session.query(SqlTraceView)
                .filter(SqlTraceView.view_id == view_id)
                .one_or_none()
            )
            if sql_view is None:
                raise MlflowException(
                    f"Trace view with ID '{view_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )
            return sql_view.to_mlflow_entity()

    def list_trace_views(self, trace_id=None, experiment_id=None):
        with self.ManagedSessionMaker() as session:
            if trace_id is not None:
                trace_info = (
                    session.query(SqlTraceInfo)
                    .filter(SqlTraceInfo.request_id == trace_id)
                    .first()
                )
                if trace_info is not None:
                    exp_id = trace_info.experiment_id
                    views = (
                        session.query(SqlTraceView)
                        .filter(
                            or_(
                                SqlTraceView.trace_id == trace_id,
                                SqlTraceView.experiment_id == exp_id,
                            )
                        )
                        .order_by(SqlTraceView.created_timestamp.desc())
                        .all()
                    )
                else:
                    views = (
                        session.query(SqlTraceView)
                        .filter(SqlTraceView.trace_id == trace_id)
                        .order_by(SqlTraceView.created_timestamp.desc())
                        .all()
                    )
            elif experiment_id is not None:
                views = (
                    session.query(SqlTraceView)
                    .filter(SqlTraceView.experiment_id == int(experiment_id))
                    .order_by(SqlTraceView.created_timestamp.desc())
                    .all()
                )
            else:
                views = []
            return [v.to_mlflow_entity() for v in views]

    def update_trace_view(self, view_id="", name=None, ranges=None):
        with self.ManagedSessionMaker() as session:
            sql_view = (
                session.query(SqlTraceView)
                .filter(SqlTraceView.view_id == view_id)
                .one_or_none()
            )
            if sql_view is None:
                raise MlflowException(
                    f"Trace view with ID '{view_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )

            if name is not None:
                sql_view.name = name

            if ranges is not None:
                # Replace all existing ranges
                sql_view.ranges.clear()
                for i, r in enumerate(ranges):
                    sql_view.ranges.append(
                        SqlTraceViewRange(
                            range_id=r.range_id or f"tvr-{uuid.uuid4().hex[:12]}",
                            view_id=sql_view.view_id,
                            position=i,
                            label=r.label,
                            description=r.description,
                            from_selector=r.from_selector.to_json(),
                            to_selector=r.to_selector.to_json() if r.to_selector else None,
                            input_path=r.input_path,
                            output_path=r.output_path,
                        )
                    )

            sql_view.last_updated_timestamp = get_current_time_millis()
            session.flush()
            return sql_view.to_mlflow_entity()

    def delete_trace_view(self, trace_id=None, experiment_id=None, view_id=""):
        with self.ManagedSessionMaker() as session:
            sql_view = (
                session.query(SqlTraceView)
                .filter(SqlTraceView.view_id == view_id)
                .one_or_none()
            )
            if sql_view is None:
                raise MlflowException(
                    f"Trace view with ID '{view_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(sql_view)
```

Note: Add `SqlTraceViewRange` to the imports from `dbmodels.models` at the top of the file. Find the existing `SqlTraceView` import and add `SqlTraceViewRange` next to it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/store/tracking/test_trace_views.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add mlflow/store/tracking/abstract_store.py mlflow/store/tracking/sqlalchemy_store.py tests/store/tracking/test_trace_views.py
git commit -s -m "refactor: update store layer for SpanRange-based trace views

Updates abstract and SQLAlchemy store CRUD methods. Update now replaces
ranges wholesale. Cascade delete handles range cleanup.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: API Layer — Handlers, REST Store, Service Client, MlflowClient

**Files:**
- Modify: `mlflow/server/handlers.py:4187-4320`
- Modify: `mlflow/store/tracking/rest_store.py:991-1066`
- Modify: `mlflow/tracking/_tracking_service/client.py:1132-1166`
- Modify: `mlflow/tracking/client.py:6713-6771`

- [ ] **Step 1: Update handlers.py**

Replace the create and update handlers (lines 4187-4256) in `mlflow/server/handlers.py`:

```python
@_disable_if_artifacts_only
def _create_trace_view(trace_id=None, experiment_id=None):
    from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView

    body = request.get_json(force=True)
    ranges_data = body.get("ranges", [])
    ranges = []
    for i, rd in enumerate(ranges_data):
        from_sel = SpanSelector.from_dict(rd["from_selector"])
        to_sel_data = rd.get("to_selector")
        to_sel = SpanSelector.from_dict(to_sel_data) if to_sel_data else None
        ranges.append(
            SpanRange(
                from_selector=from_sel,
                to_selector=to_sel,
                label=rd.get("label", ""),
                description=rd.get("description", ""),
                input_path=rd.get("input_path"),
                output_path=rd.get("output_path"),
                position=i,
            )
        )
    view = TraceView(
        name=body["name"],
        trace_id=trace_id,
        experiment_id=experiment_id,
        ranges=ranges,
        created_by=body.get("created_by"),
    )
    created_view = _get_tracking_store().create_trace_view(view)
    return jsonify({"trace_view": created_view.to_dict()})


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_trace_view(trace_id, view_id):
    view = _get_tracking_store().get_trace_view(trace_id, view_id)
    return jsonify({"trace_view": view.to_dict()})


@catch_mlflow_exception
@_disable_if_artifacts_only
def _list_trace_views(trace_id=None, experiment_id=None):
    views = _get_tracking_store().list_trace_views(
        trace_id=trace_id, experiment_id=experiment_id
    )
    return jsonify({"trace_views": [v.to_dict() for v in views]})


@catch_mlflow_exception
@_disable_if_artifacts_only
def _update_trace_view(trace_id=None, experiment_id=None, view_id=None):
    from mlflow.entities.trace_view import SpanRange, SpanSelector

    body = request.get_json(force=True)
    kwargs = {}
    if "name" in body:
        kwargs["name"] = body["name"]
    if "ranges" in body:
        ranges = []
        for i, rd in enumerate(body["ranges"]):
            from_sel = SpanSelector.from_dict(rd["from_selector"])
            to_sel_data = rd.get("to_selector")
            to_sel = SpanSelector.from_dict(to_sel_data) if to_sel_data else None
            ranges.append(
                SpanRange(
                    from_selector=from_sel,
                    to_selector=to_sel,
                    label=rd.get("label", ""),
                    description=rd.get("description", ""),
                    input_path=rd.get("input_path"),
                    output_path=rd.get("output_path"),
                    position=i,
                )
            )
        kwargs["ranges"] = ranges

    updated_view = _get_tracking_store().update_trace_view(
        view_id=view_id,
        **kwargs,
    )
    return jsonify({"trace_view": updated_view.to_dict()})


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_trace_view(trace_id=None, experiment_id=None, view_id=None):
    _get_tracking_store().delete_trace_view(
        trace_id=trace_id, experiment_id=experiment_id, view_id=view_id
    )
    return jsonify({})
```

- [ ] **Step 2: Update rest_store.py**

Replace the view methods (lines 991-1066) in `mlflow/store/tracking/rest_store.py`:

```python
    def _get_trace_view_base_path(self, trace_id=None, experiment_id=None):
        if trace_id is not None:
            return f"/ajax-api/2.0/mlflow/traces/{trace_id}/views"
        return f"/ajax-api/2.0/mlflow/experiments/{experiment_id}/views"

    def create_trace_view(self, view):
        from mlflow.entities.trace_view import TraceView

        base = self._get_trace_view_base_path(
            trace_id=view.trace_id, experiment_id=view.experiment_id
        )
        body = {
            "name": view.name,
            "created_by": view.created_by,
            "ranges": [r.to_dict() for r in view.ranges],
        }
        resp = self._call_json_api(base, "POST", json_body=body)
        return TraceView.from_dict(resp["trace_view"])

    def get_trace_view(self, trace_id, view_id):
        from mlflow.entities.trace_view import TraceView

        endpoint = f"/ajax-api/2.0/mlflow/traces/{trace_id}/views/{view_id}"
        resp = self._call_json_api(endpoint, "GET")
        return TraceView.from_dict(resp["trace_view"])

    def list_trace_views(self, trace_id=None, experiment_id=None):
        from mlflow.entities.trace_view import TraceView

        base = self._get_trace_view_base_path(
            trace_id=trace_id, experiment_id=experiment_id
        )
        resp = self._call_json_api(base, "GET")
        return [TraceView.from_dict(v) for v in resp.get("trace_views", [])]

    def update_trace_view(self, view_id="", name=None, ranges=None):
        from mlflow.entities.trace_view import TraceView

        # We need trace_id or experiment_id for the URL path.
        # Get the view first to determine its scope.
        # For REST store, update is called with view_id only.
        # The handlers accept view_id as a URL param, so we use a trace-scoped path.
        endpoint = f"/ajax-api/2.0/mlflow/traces/_/views/{view_id}"
        body = {}
        if name is not None:
            body["name"] = name
        if ranges is not None:
            body["ranges"] = [r.to_dict() for r in ranges]
        resp = self._call_json_api(endpoint, "PATCH", json_body=body)
        return TraceView.from_dict(resp["trace_view"])

    def delete_trace_view(self, trace_id=None, experiment_id=None, view_id=""):
        base = self._get_trace_view_base_path(
            trace_id=trace_id, experiment_id=experiment_id
        )
        endpoint = f"{base}/{view_id}"
        self._call_json_api(endpoint, "DELETE")
```

- [ ] **Step 3: Update _tracking_service/client.py**

Replace the view methods (lines 1132-1166) in `mlflow/tracking/_tracking_service/client.py`:

```python
    def create_trace_view(self, view):
        return self.store.create_trace_view(view)

    def get_trace_view(self, trace_id, view_id):
        return self.store.get_trace_view(trace_id, view_id)

    def list_trace_views(self, trace_id=None, experiment_id=None):
        return self.store.list_trace_views(trace_id=trace_id, experiment_id=experiment_id)

    def update_trace_view(self, view_id="", name=None, ranges=None):
        return self.store.update_trace_view(
            view_id=view_id,
            name=name,
            ranges=ranges,
        )

    def delete_trace_view(self, trace_id=None, experiment_id=None, view_id=""):
        return self.store.delete_trace_view(
            trace_id=trace_id, experiment_id=experiment_id, view_id=view_id
        )
```

- [ ] **Step 4: Update tracking/client.py**

Replace the view methods (lines 6713-6771) in `mlflow/tracking/client.py`:

```python
    def create_trace_view(
        self,
        trace_id=None,
        experiment_id=None,
        name="",
        ranges=None,
        created_by=None,
    ):
        from mlflow.entities.trace_view import TraceView

        view = TraceView(
            name=name,
            trace_id=trace_id,
            experiment_id=experiment_id,
            ranges=ranges or [],
            created_by=created_by,
        )
        return self._tracking_client.create_trace_view(view)

    def list_trace_views(self, trace_id=None, experiment_id=None):
        return self._tracking_client.list_trace_views(
            trace_id=trace_id, experiment_id=experiment_id
        )

    def get_trace_view(self, trace_id, view_id):
        return self._tracking_client.get_trace_view(trace_id, view_id)

    def update_trace_view(self, view_id="", name=None, ranges=None):
        return self._tracking_client.update_trace_view(
            view_id=view_id,
            name=name,
            ranges=ranges,
        )

    def delete_trace_view(self, trace_id=None, experiment_id=None, view_id=""):
        return self._tracking_client.delete_trace_view(
            trace_id=trace_id, experiment_id=experiment_id, view_id=view_id
        )
```

- [ ] **Step 5: Run store tests to verify full stack**

Run: `uv run --no-sync pytest tests/store/tracking/test_trace_views.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add mlflow/server/handlers.py mlflow/store/tracking/rest_store.py mlflow/tracking/_tracking_service/client.py mlflow/tracking/client.py
git commit -s -m "refactor: update API layer for SpanRange-based trace views

Updates handlers, REST store, service client, and MlflowClient to use
ranges payload instead of span_filter/input_path/output_path/description.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Workspace Store

**Files:**
- Modify: `mlflow/store/tracking/sqlalchemy_workspace_store.py:496-695`

- [ ] **Step 1: Update workspace store view methods**

In `mlflow/store/tracking/sqlalchemy_workspace_store.py`, update the `update_trace_view` method (lines 594-652). The signature changes from individual filter params to `view_id`, `name`, `ranges`. The `get_trace_view`, `list_trace_views`, and `delete_trace_view` methods don't need signature changes — only `update_trace_view` does.

Replace the `update_trace_view` method:

```python
    def update_trace_view(self, view_id="", name=None, ranges=None):
        with self.ManagedSessionMaker() as session:
            workspace_experiment_ids = (
                session
                .query(SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace == self._get_active_workspace())
                .subquery()
            )

            sql_view = (
                session.query(SqlTraceView)
                .filter(SqlTraceView.view_id == view_id)
                .filter(
                    or_(
                        SqlTraceView.experiment_id.in_(
                            select(workspace_experiment_ids.c.experiment_id)
                        ),
                        SqlTraceView.trace_id.in_(
                            session.query(SqlTraceInfo.request_id)
                            .filter(
                                SqlTraceInfo.experiment_id.in_(
                                    select(workspace_experiment_ids.c.experiment_id)
                                )
                            )
                        ),
                    )
                )
                .one_or_none()
            )
            if sql_view is None:
                raise MlflowException(
                    f"Trace view with ID '{view_id}' not found",
                    RESOURCE_DOES_NOT_EXIST,
                )

            if name is not None:
                sql_view.name = name

            if ranges is not None:
                sql_view.ranges.clear()
                for i, r in enumerate(ranges):
                    sql_view.ranges.append(
                        SqlTraceViewRange(
                            range_id=r.range_id or f"tvr-{uuid.uuid4().hex[:12]}",
                            view_id=sql_view.view_id,
                            position=i,
                            label=r.label,
                            description=r.description,
                            from_selector=r.from_selector.to_json(),
                            to_selector=r.to_selector.to_json() if r.to_selector else None,
                            input_path=r.input_path,
                            output_path=r.output_path,
                        )
                    )

            sql_view.last_updated_timestamp = get_current_time_millis()
            session.flush()
            return sql_view.to_mlflow_entity()
```

Note: Add `SqlTraceViewRange` to the imports from `dbmodels.models` at the top of the file, and `uuid` to stdlib imports.

- [ ] **Step 2: Run store tests**

Run: `uv run --no-sync pytest tests/store/tracking/test_trace_views.py -v`
Expected: All PASS (workspace store inherits from SqlAlchemyStore, so the base tests validate it).

- [ ] **Step 3: Commit**

```bash
git add mlflow/store/tracking/sqlalchemy_workspace_store.py
git commit -s -m "refactor: update workspace store for SpanRange-based trace views

Updates update_trace_view signature and implementation to use ranges
instead of individual filter/path/description params.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: DFS Resolution — view_utils.py

**Files:**
- Modify: `mlflow/tracing/utils/view_utils.py`
- Test: `tests/tracing/utils/test_view_utils.py`

- [ ] **Step 1: Write failing tests for DFS resolution**

Replace the entire contents of `tests/tracing/utils/test_view_utils.py`:

```python
from __future__ import annotations

import json

import pytest

from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.tracing.utils.view_utils import (
    apply_jsonpath,
    resolve_range,
    resolve_view,
    validate_jsonpath,
)


def _make_span(name, span_type, span_id, inputs=None, outputs=None, children=None):
    return {
        "name": name,
        "span_type": span_type,
        "context": {"span_id": span_id},
        "inputs": json.dumps(inputs or {}),
        "outputs": json.dumps(outputs or {}),
        "attributes": {},
        "child_spans": children or [],
    }


def _build_trace():
    """Build a tree matching the prototype demo trace."""
    tokenizer = _make_span("tokenizer", "INTERNAL", "s2a", outputs={"tokens": 150})
    llm1 = _make_span(
        "ChatOpenAI", "LLM", "s2",
        outputs={"reasoning": "Need to search", "tool_call": "search"},
        children=[tokenizer],
    )
    sql = _make_span("sql_query", "INTERNAL", "s3a", inputs={"sql": "SELECT *"}, outputs={"rows": 8})
    tool1 = _make_span(
        "search_content", "TOOL", "s3",
        inputs={"query": "template"},
        outputs={"results": [{"title": "Welcome"}], "count": 8},
        children=[sql],
    )
    llm2 = _make_span("ChatOpenAI", "LLM", "s4", outputs={"reasoning": "Found template"})
    tool2 = _make_span(
        "get_transaction", "TOOL", "s5",
        outputs={"transaction": {"buyer": "John"}},
    )
    llm3 = _make_span("ChatOpenAI", "LLM", "s6", outputs={"reasoning": "Need participants"})
    tool3 = _make_span("list_participants", "TOOL", "s7", outputs={"participants": []})
    llm4 = _make_span(
        "ChatOpenAI", "LLM", "s8",
        outputs={"reasoning": "Must escalate", "decision": "escalate"},
    )
    root = _make_span(
        "AgentExecutor", "CHAIN", "s1",
        inputs={"task": "send email"},
        outputs={"status": "escalated"},
        children=[llm1, tool1, llm2, tool2, llm3, tool3, llm4],
    )
    return root


class TestResolveRange:
    def test_single_span_by_id(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_id="s8"))
        matched = resolve_range(root, r)
        assert len(matched) == 1
        assert matched[0]["context"]["span_id"] == "s8"

    def test_span_with_subtree(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_id="s2"))
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s2", "s2a"]

    def test_range_between_two_spans(self):
        root = _build_trace()
        r = SpanRange(
            from_selector=SpanSelector(span_id="s2"),
            to_selector=SpanSelector(span_name="search_content"),
        )
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s2", "s2a", "s3", "s3a"]

    def test_range_by_type(self):
        root = _build_trace()
        r = SpanRange(
            from_selector=SpanSelector(span_id="s4"),
            to_selector=SpanSelector(span_name="get_transaction"),
        )
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s4", "s5"]

    def test_root_span_matches_all(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_name="AgentExecutor"))
        matched = resolve_range(root, r)
        assert len(matched) == 10

    def test_no_match_returns_empty(self):
        root = _build_trace()
        r = SpanRange(from_selector=SpanSelector(span_name="nonexistent"))
        matched = resolve_range(root, r)
        assert matched == []

    def test_to_selector_not_found_falls_back_to_subtree(self):
        root = _build_trace()
        r = SpanRange(
            from_selector=SpanSelector(span_id="s2"),
            to_selector=SpanSelector(span_name="nonexistent"),
        )
        matched = resolve_range(root, r)
        ids = [s["context"]["span_id"] for s in matched]
        assert ids == ["s2", "s2a"]


class TestResolveView:
    def test_full_view_resolution(self):
        root = _build_trace()
        view = TraceView(
            name="test",
            trace_id="t1",
            ranges=[
                SpanRange(
                    from_selector=SpanSelector(span_name="AgentExecutor"),
                    label="Summary",
                    description="Overview",
                    position=0,
                ),
                SpanRange(
                    from_selector=SpanSelector(span_id="s2"),
                    to_selector=SpanSelector(span_name="search_content"),
                    label="Template Lookup",
                    description="Searched for template",
                    input_path="$.reasoning",
                    output_path="$.results",
                    position=1,
                ),
                SpanRange(
                    from_selector=SpanSelector(span_id="s8"),
                    label="Escalation",
                    description="Agent escalated",
                    output_path="$.reasoning",
                    position=2,
                ),
            ],
        )
        results = resolve_view(root, view)
        assert len(results) == 3

        assert results[0]["label"] == "Summary"
        assert len(results[0]["spans"]) == 10

        assert results[1]["label"] == "Template Lookup"
        assert len(results[1]["spans"]) == 4
        assert results[1]["extracted_input"] == "Need to search"
        assert results[1]["extracted_output"] == '[{"title": "Welcome"}]'

        assert results[2]["label"] == "Escalation"
        assert len(results[2]["spans"]) == 1
        assert results[2]["extracted_output"] == "Must escalate"


class TestApplyJsonpath:
    def test_simple_path(self):
        data = json.dumps({"reasoning": "hello"})
        result, success = apply_jsonpath(data, "$.reasoning")
        assert success
        assert result == "hello"

    def test_no_match(self):
        data = json.dumps({"other": "value"})
        result, success = apply_jsonpath(data, "$.nonexistent")
        assert not success

    def test_none_expr(self):
        result, success = apply_jsonpath('{"a": 1}', None)
        assert not success


class TestValidateJsonpath:
    def test_valid(self):
        ok, err = validate_jsonpath("$.field")
        assert ok
        assert err is None

    def test_empty_is_valid(self):
        ok, err = validate_jsonpath("")
        assert ok

    def test_invalid(self):
        ok, err = validate_jsonpath("[invalid")
        assert not ok
        assert err is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/tracing/utils/test_view_utils.py -v`
Expected: FAIL — `resolve_range` and `resolve_view` do not exist yet.

- [ ] **Step 3: Implement DFS resolution**

Replace the entire contents of `mlflow/tracing/utils/view_utils.py`:

```python
from __future__ import annotations

import json
import logging

from mlflow.entities.trace_view import SpanSelector, TraceView

try:
    from jsonpath_ng import parse as jsonpath_parse

    _HAS_JSONPATH = True
except ImportError:
    _HAS_JSONPATH = False

_logger = logging.getLogger(__name__)


def _unwrap_json_str(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _matches_span_type(span: dict, span_type: str) -> bool:
    if "span_type" in span and span["span_type"] == span_type:
        return True
    attrs = span.get("attributes", {})
    attr_type = attrs.get("mlflow.spanType")
    if attr_type is not None:
        return _unwrap_json_str(attr_type) == span_type
    return False


def _get_span_id(span: dict) -> str | None:
    ctx = span.get("context")
    if isinstance(ctx, dict):
        return ctx.get("span_id")
    return None


def _span_matches(span: dict, selector: SpanSelector) -> bool:
    if selector.span_name is not None and span.get("name") != selector.span_name:
        return False
    if selector.span_type is not None and not _matches_span_type(span, selector.span_type):
        return False
    if selector.span_id is not None and _get_span_id(span) != selector.span_id:
        return False
    if selector.attribute_key is not None:
        attrs = span.get("attributes", {})
        if selector.attribute_key not in attrs:
            return False
        if selector.attribute_value is not None:
            if str(attrs[selector.attribute_key]) != str(selector.attribute_value):
                return False
    return True


def _dfs_order(root: dict) -> list[dict]:
    result = []
    stack = [root]
    while stack:
        span = stack.pop()
        result.append(span)
        children = span.get("child_spans", [])
        stack.extend(reversed(children))
    return result


def _subtree_ids(span: dict) -> set[str]:
    ids = set()
    stack = [span]
    while stack:
        s = stack.pop()
        sid = _get_span_id(s)
        if sid:
            ids.add(sid)
        stack.extend(s.get("child_spans", []))
    return ids


def resolve_range(root: dict, span_range) -> list[dict]:
    spans_dfs = _dfs_order(root)

    # Find from span
    from_idx = next(
        (i for i, s in enumerate(spans_dfs) if _span_matches(s, span_range.from_selector)),
        None,
    )
    if from_idx is None:
        return []

    from_span = spans_dfs[from_idx]

    # No to_selector: return from span + subtree
    if span_range.to_selector is None:
        sub_ids = _subtree_ids(from_span)
        return [s for s in spans_dfs[from_idx:] if _get_span_id(s) in sub_ids]

    # Find to span after from
    to_idx = next(
        (j for j in range(from_idx + 1, len(spans_dfs))
         if _span_matches(spans_dfs[j], span_range.to_selector)),
        None,
    )
    if to_idx is None:
        sub_ids = _subtree_ids(from_span)
        return [s for s in spans_dfs[from_idx:] if _get_span_id(s) in sub_ids]

    to_span = spans_dfs[to_idx]

    # Collect from from_idx through end of to_span's subtree
    to_sub_ids = _subtree_ids(to_span)
    end_idx = to_idx
    for k in range(to_idx + 1, len(spans_dfs)):
        if _get_span_id(spans_dfs[k]) in to_sub_ids:
            end_idx = k
        else:
            break

    return spans_dfs[from_idx:end_idx + 1]


def _extract_io(span: dict) -> tuple[str | None, str | None]:
    inputs = span.get("inputs")
    if inputs is None:
        raw = span.get("attributes", {}).get("mlflow.spanInputs")
        if raw is not None:
            inputs = _unwrap_json_str(raw)

    outputs = span.get("outputs")
    if outputs is None:
        raw = span.get("attributes", {}).get("mlflow.spanOutputs")
        if raw is not None:
            outputs = _unwrap_json_str(raw)

    def _serialize(val):
        if val is None:
            return None
        if isinstance(val, str):
            return val
        return json.dumps(val)

    return _serialize(inputs), _serialize(outputs)


def apply_jsonpath(data: str, jsonpath_expr: str | None) -> tuple[str | None, bool]:
    if not jsonpath_expr:
        return None, False
    if not _HAS_JSONPATH:
        _logger.warning("jsonpath-ng is not installed. Install it with: pip install jsonpath-ng")
        return None, False
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, False

    try:
        expr = jsonpath_parse(jsonpath_expr)
    except Exception:
        return None, False

    matches = expr.find(parsed)
    if not matches:
        return None, False

    values = [m.value for m in matches if m.value is not None]
    if not values:
        return None, False

    parts = [str(v) for v in values]
    result = "\n".join(parts)
    return result, True


def validate_jsonpath(expr: str) -> tuple[bool, str | None]:
    if not expr:
        return True, None
    if not _HAS_JSONPATH:
        return False, "jsonpath-ng is not installed"
    try:
        jsonpath_parse(expr)
        return True, None
    except Exception as e:
        return False, str(e)


def resolve_view(root: dict, view: TraceView) -> list[dict]:
    results = []
    for span_range in sorted(view.ranges, key=lambda r: r.position):
        matched = resolve_range(root, span_range)

        extracted_input = None
        extracted_output = None

        if matched and span_range.input_path:
            inp_data, _ = _extract_io(matched[0])
            if inp_data:
                extracted, success = apply_jsonpath(inp_data, span_range.input_path)
                if success:
                    extracted_input = extracted

            # Fallback: try extracting from outputs of first span
            if extracted_input is None:
                _, out_data = _extract_io(matched[0])
                if out_data:
                    extracted, success = apply_jsonpath(out_data, span_range.input_path)
                    if success:
                        extracted_input = extracted

        if matched and span_range.output_path:
            _, out_data = _extract_io(matched[-1])
            if out_data:
                extracted, success = apply_jsonpath(out_data, span_range.output_path)
                if success:
                    extracted_output = extracted

        results.append({
            "label": span_range.label,
            "description": span_range.description,
            "spans": matched,
            "extracted_input": extracted_input,
            "extracted_output": extracted_output,
        })
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/tracing/utils/test_view_utils.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mlflow/tracing/utils/view_utils.py tests/tracing/utils/test_view_utils.py
git commit -s -m "refactor: replace apply_view with DFS-based resolve_range/resolve_view

Implements DFS tree walk with from/to selector matching, subtree
collection, and per-range JSONPath extraction.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Trace Entity Convenience Methods

**Files:**
- Modify: `mlflow/entities/trace.py:298-366`

- [ ] **Step 1: Update create_view and summarize/analyze methods**

In `mlflow/entities/trace.py`, replace lines 298-366:

```python
    def create_view(self, name, ranges=None, created_by=None):
        from mlflow.tracking import MlflowClient

        return MlflowClient().create_trace_view(
            trace_id=self.info.trace_id,
            name=name,
            ranges=ranges or [],
            created_by=created_by,
        )

    @property
    def views(self):
        from mlflow.tracking import MlflowClient

        return MlflowClient().list_trace_views(trace_id=self.info.trace_id)

    def delete_view(self, view_id):
        from mlflow.tracking import MlflowClient

        MlflowClient().delete_trace_view(trace_id=self.info.trace_id, view_id=view_id)

    def summarize(self, model="openai:/gpt-4o-mini", view=None):
        from mlflow.genai.judges.utils.invocation_utils import invoke_judge_model

        prompt = "Summarize this trace concisely. Focus on what the agent did, key decisions, and the outcome."
        if view:
            from mlflow.tracing.utils.view_utils import resolve_view

            root_span = self.data.spans[0].to_dict() if self.data.spans else None
            if root_span:
                results = resolve_view(root_span, view)
                summary_parts = []
                for r in results:
                    summary_parts.append(f"**{r['label']}**: {r['description']}")
                    if r["extracted_input"]:
                        summary_parts.append(f"  Input: {r['extracted_input']}")
                    if r["extracted_output"]:
                        summary_parts.append(f"  Output: {r['extracted_output']}")
                prompt += "\n\nView summary:\n" + "\n".join(summary_parts)
        feedback = invoke_judge_model(
            model_uri=model,
            prompt=prompt,
            assessment_name="trace_summary",
            trace=self,
        )
        return feedback.value

    def analyze(self, question, model="openai:/gpt-4o-mini", view=None):
        from mlflow.genai.judges.utils.invocation_utils import invoke_judge_model

        prompt = question
        if view:
            from mlflow.tracing.utils.view_utils import resolve_view

            root_span = self.data.spans[0].to_dict() if self.data.spans else None
            if root_span:
                results = resolve_view(root_span, view)
                summary_parts = []
                for r in results:
                    summary_parts.append(f"**{r['label']}**: {r['description']}")
                    if r["extracted_input"]:
                        summary_parts.append(f"  Input: {r['extracted_input']}")
                    if r["extracted_output"]:
                        summary_parts.append(f"  Output: {r['extracted_output']}")
                prompt += "\n\nView summary:\n" + "\n".join(summary_parts)
        feedback = invoke_judge_model(
            model_uri=model,
            prompt=prompt,
            assessment_name="trace_analysis",
            trace=self,
        )
        return feedback.value
```

- [ ] **Step 2: Commit**

```bash
git add mlflow/entities/trace.py
git commit -s -m "refactor: update Trace convenience methods for SpanRange model

Updates create_view, summarize, and analyze to use ranges and
resolve_view instead of the old apply_view with single filter.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Clean Up References to SpanFilter

**Files:**
- Modify: any remaining references to `SpanFilter` across the codebase

- [ ] **Step 1: Search for remaining SpanFilter references**

Run: `grep -r "SpanFilter" mlflow/ tests/ --include="*.py" -l`

This will find any files still importing or referencing the old `SpanFilter` class. The entity module itself no longer exports it, so any remaining references will cause import errors.

- [ ] **Step 2: Fix each reference**

For each file found, update `SpanFilter` → `SpanSelector` and update any related usage (e.g., constructing views with the old `span_filter=` parameter).

Common locations to check:
- `mlflow/demo/generators/traces.py` — demo data generators that create example views
- `mlflow/cli/traces.py` — CLI commands that create views
- Any assistant skill instructions that reference `SpanFilter`
- `tests/entities/test_trace_view_methods.py` — if it exists

- [ ] **Step 3: Run all trace-related tests**

Run: `uv run --no-sync pytest tests/entities/test_trace_view.py tests/store/tracking/test_trace_views.py tests/tracing/utils/test_view_utils.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -s -m "refactor: clean up remaining SpanFilter references

Replaces all remaining SpanFilter imports and usages with SpanSelector
across demo generators, CLI, and tests.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Delete Stale SQLite DB and Verify Fresh Start

Since the migration was rewritten, any existing local SQLite databases will have the old schema.

- [ ] **Step 1: Delete existing test/dev databases**

```bash
rm -f mlflow/mlflow.db
rm -f /tmp/mlflow*.db
```

- [ ] **Step 2: Start dev server and verify**

```bash
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &
tail -f /tmp/mlflow-dev-server.log
```

Verify the server starts without migration errors. Check `http://localhost:5000` responds.

- [ ] **Step 3: Create a test view via Python**

```python
import mlflow
from mlflow.entities.trace_view import SpanRange, SpanSelector

client = mlflow.MlflowClient()
# Create a trace-scoped view (requires an existing trace)
# Or create an experiment-scoped view:
view = client.create_trace_view(
    experiment_id="0",
    name="Test View",
    ranges=[
        SpanRange(
            from_selector=SpanSelector(span_type="LLM"),
            label="LLM Calls",
            description="All LLM invocations",
        ),
    ],
    created_by="test",
)
print(f"Created view: {view.view_id}")
print(f"Ranges: {len(view.ranges)}")
```

- [ ] **Step 4: Commit (if any cleanup was needed)**

No commit needed unless files were changed in this task.
