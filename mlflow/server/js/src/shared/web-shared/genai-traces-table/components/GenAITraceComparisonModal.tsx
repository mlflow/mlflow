import { ModelTraceExplorerSummaryView } from '../../model-trace-explorer/summary-view/ModelTraceExplorerSummaryView';
import { ModelTraceExplorerViewStateProvider } from '../../model-trace-explorer/ModelTraceExplorerViewStateContext';
import type { ModelTrace } from '../../model-trace-explorer';
import type { RunEvaluationTracesDataEntry, TraceActions, TraceInfoV3 } from './../types';
import { useState, useEffect } from 'react';

import { Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { useIntl } from '@databricks/i18n';

export const TraceComparisonModal = ({
  traces,
  onClose,
  getTrace,
}: {
  traces: RunEvaluationTracesDataEntry[];
  onClose: () => void;
  getTrace?: (requestId?: string, traceId?: string) => Promise<ModelTrace | undefined>;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [modelTraces, setModelTraces] = useState<ModelTrace[]>([]);

  useEffect(() => {
    let cancelled = false;
    const fetchTraces = async () => {
      if (!getTrace) {
        setModelTraces([]);
        return;
      }
      const results = await Promise.all(traces.map((trace) => getTrace(trace.evaluationId, trace.traceInfo?.trace_id)));
      if (!cancelled) {
        setModelTraces(results.filter((t): t is ModelTrace => Boolean(t)));
      }
    };
    fetchTraces();
    return () => {
      cancelled = true;
    };
  }, [traces, getTrace]);

  return (
    <Modal
      componentId="mlflow.genai-traces-table.compare-modal"
      visible
      title={intl.formatMessage({ defaultMessage: 'Compare traces', description: 'Compare traces modal title' })}
      onCancel={onClose}
      size="wide"
      verticalSizing="maxed_out"
      footer={null}
      css={{ width: '100% !important' }}
    >
      <div css={{ display: 'flex', gap: theme.spacing.lg, overflow: 'auto', height: '100%' }}>
        {modelTraces.length === 0 ? (
          <Typography.Text>
            {intl.formatMessage({ defaultMessage: 'Loading traces…', description: 'Loading traces message' })}
          </Typography.Text>
        ) : (
          modelTraces.map((modelTrace, index) => (
            <div key={traces[index].evaluationId} css={{ flex: 1, minWidth: 400, height: '100%', overflow: 'auto' }}>
              <ModelTraceExplorerViewStateProvider
                modelTrace={modelTrace}
                initialActiveView="summary"
                assessmentsPaneEnabled
              >
                <ModelTraceExplorerSummaryView modelTrace={modelTrace} />
              </ModelTraceExplorerViewStateProvider>
            </div>
          ))
        )}
      </div>
    </Modal>
  );
};
