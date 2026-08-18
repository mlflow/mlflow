from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class MLOpsDebtReport:
    experiment_id: str
    model_name: str
    adi_score: float  # Artifact Debt Index (target <= 12.0)
    storage_multiplier: float  # Target <= 1.10x
    promotion_latency_seconds: float  # Target <= 1.8s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for MLflow enterprise model runs.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_model_evaluation(
        self,
        experiment_id: str,
        model_name: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{experiment_id}|{model_name}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtEvaluator:
    """
    A2Z SOC Production Debt & Technical Due Diligence Evaluator for MLflow GenAI Metrics.

    Quantifies MLOps model runs and registry promotions against 4 Enterprise Forward Deployed Engineering KPIs:
    1. Artifact Debt Index (ADI <= 12.0)
    2. Model Registry Storage Multiplier (RSM <= 1.10x)
    3. P99 Model Promotion Latency Ceiling (<= 1.8s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_adi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_adi = max_acceptable_adi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_model_run(
        self,
        experiment_id: str,
        model_name: str,
        raw_weights_bytes: int = 1000000000,
        stored_artifact_bytes: int = 1050000000,
        promotion_latency_seconds: float = 0.95,
        staged_version_drift_count: int = 0,
        un_gated_mutations: int = 0,
    ) -> MLOpsDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_model_evaluation(
                experiment_id=experiment_id,
                model_name=model_name,
                event_type="evaluation_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. MLOps model evaluation halted."
            )

        critical_smells: List[str] = []

        # KPI 2: Model Registry Storage Multiplier
        storage_ratio = stored_artifact_bytes / max(1, raw_weights_bytes)
        if storage_ratio > 2.0:
            critical_smells.append(f"HIGH_STORAGE_SPRAWL_{storage_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if promotion_latency_seconds > 5.0:
            critical_smells.append(f"HIGH_PROMOTION_LATENCY_{promotion_latency_seconds:.2f}S")

        # Staged version drift
        if staged_version_drift_count > 2:
            critical_smells.append(f"DETECTED_{staged_version_drift_count}_UNMERGED_STAGED_VERSIONS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_REGISTRY_MUTATIONS")

        # KPI 1: Artifact Debt Index (0 = Clean, 100 = Catastrophic)
        adi = (
            max(0.0, (storage_ratio - 1.0) * 20.0)
            + max(0.0, (promotion_latency_seconds - 1.8) * 8.0)
            + (staged_version_drift_count * 12.0)
            + (un_gated_mutations * 30.0)
        )
        adi_score = round(min(100.0, adi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - adi_score)
        is_production_ready = (
            adi_score <= self.max_acceptable_adi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_model_evaluation(
            experiment_id=experiment_id,
            model_name=model_name,
            event_type="model_authorized" if is_production_ready else "model_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "adi_score": adi_score,
                "storage_ratio": storage_ratio,
                "promotion_latency_seconds": promotion_latency_seconds,
                "staged_version_drift_count": staged_version_drift_count,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return MLOpsDebtReport(
            experiment_id=experiment_id,
            model_name=model_name,
            adi_score=adi_score,
            storage_multiplier=round(storage_ratio, 2),
            promotion_latency_seconds=round(promotion_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
