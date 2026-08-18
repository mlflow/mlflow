import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../../mlflow/metrics/genai/production_debt.py",
)
spec = importlib.util.spec_from_file_location("mlflow_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["mlflow_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtEvaluator = production_debt_mod.ProductionDebtEvaluator
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ProductionDebtEvaluator(
            never_equate_intent_to_approval=True,
            max_acceptable_adi=12.0,
        )

    def test_clean_model_run_passes_readiness(self) -> None:
        report = self.evaluator.evaluate_model_run(
            experiment_id="exp_enterprise_fraud",
            model_name="fraud_detection_v4",
            raw_weights_bytes=1000000000,
            stored_artifact_bytes=1040000000,
            promotion_latency_seconds=0.85,
            staged_version_drift_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.adi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_model_run_fails_debt(self) -> None:
        report = self.evaluator.evaluate_model_run(
            experiment_id="exp_runaway_artifacts",
            model_name="unoptimized_llm_finetune",
            raw_weights_bytes=1000000000,
            stored_artifact_bytes=3800000000,  # High storage sprawl (3.8x)
            promotion_latency_seconds=8.5,  # High latency
            staged_version_drift_count=4,  # 4 unmerged versions
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.adi_score, 50.0)
        self.assertIn("HIGH_STORAGE_SPRAWL_3.80X", report.critical_smells)
        self.assertIn("HIGH_PROMOTION_LATENCY_8.50S", report.critical_smells)
        self.assertIn("DETECTED_4_UNMERGED_STAGED_VERSIONS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_REGISTRY_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.evaluator.evaluate_model_run("e-1", "m-1")
        self.evaluator.evaluate_model_run("e-2", "m-2")
        self.evaluator.evaluate_model_run("e-3", "m-3")

        entries = self.evaluator.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.evaluator.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
