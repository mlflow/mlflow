import type { CellContext, ColumnDefTemplate } from '@tanstack/react-table';
import React from 'react';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { LegacyTooltip } from '@databricks/design-system';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

export const TracesViewTableTimestampCell: ColumnDefTemplate<CellContext<ModelTraceInfoWithRunName, unknown>> =
  React.memo(
    ({ row: { original } }) => {
      if (!original.timestamp_ms) {
        return null;
      }
      return (
        <LegacyTooltip
          title={new Date(original.timestamp_ms).toLocaleString(navigator.language, {
            timeZoneName: 'short',
          })}
          placement="right"
        >
          <span>{Utils.timeSinceStr(original.timestamp_ms)}</span>
        </LegacyTooltip>
      );
    },
    () => true,
  );
