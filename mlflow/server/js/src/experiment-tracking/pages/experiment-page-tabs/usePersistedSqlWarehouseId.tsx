import { useCallback, useEffect, useState } from 'react';
import { getLocalStorageItem, setLocalStorageItem } from '@databricks/web-shared/hooks';

const PER_EXPERIMENT_KEY_PREFIX = 'mlflow_warehouse_experiment_';
const PER_EXPERIMENT_KEY_VERSION = 1;

interface PerExperimentWarehouseValue {
  SQL_WAREHOUSE?: { id: string; timestamp: number };
}

function readPerExperimentWarehouseId(experimentId: string): string | undefined {
  const value = getLocalStorageItem<PerExperimentWarehouseValue | null>(
    `${PER_EXPERIMENT_KEY_PREFIX}${experimentId}`,
    PER_EXPERIMENT_KEY_VERSION,
    true,
    null,
  );
  return value?.SQL_WAREHOUSE?.id ?? undefined;
}

/**
 * Reads the warehouse ID from a per-experiment localStorage key.
 * The ComputeSelectorDialogCombobox with autoSelectIfValueUnspecified
 * handles default selection when no value is stored.
 */
export const usePersistedSqlWarehouseId = (experimentId: string) => {
  const [warehouseId, setWarehouseId] = useState<string | undefined>(() => readPerExperimentWarehouseId(experimentId));

  // Re-read from localStorage when experimentId changes (component may not remount)
  useEffect(() => {
    setWarehouseId(readPerExperimentWarehouseId(experimentId));
  }, [experimentId]);

  const setSqlWarehouseId = useCallback(
    (id: string | undefined | null) => {
      const resolved = id ?? undefined;
      setWarehouseId(resolved);

      // Persist to per-experiment key
      const storageKey = `${PER_EXPERIMENT_KEY_PREFIX}${experimentId}`;
      if (resolved) {
        setLocalStorageItem<PerExperimentWarehouseValue>(storageKey, PER_EXPERIMENT_KEY_VERSION, true, {
          SQL_WAREHOUSE: { id: resolved, timestamp: Date.now() },
        });
      } else {
        setLocalStorageItem<PerExperimentWarehouseValue | null>(storageKey, PER_EXPERIMENT_KEY_VERSION, true, null);
      }
    },
    [experimentId],
  );

  return [warehouseId, setSqlWarehouseId] as const;
};
