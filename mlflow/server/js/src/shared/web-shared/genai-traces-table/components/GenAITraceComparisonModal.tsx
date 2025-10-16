import { ModelTraceExplorer } from '../../model-trace-explorer';
import type { ModelTrace } from '../../model-trace-explorer';
import type { RunEvaluationTracesDataEntry } from './../types';
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
  getTrace?: (traceId?: string) => Promise<ModelTrace | undefined>;
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
      const results = await Promise.all(
        traces.map((trace) => {
          const traceId = trace.traceInfo?.trace_id ?? trace.requestId ?? trace.evaluationId;
          return getTrace(traceId);
        }),
      );
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
      dangerouslySetAntdProps={{ width: '95%' }}
    >
      <div css={{ height: '100%', overflow: 'auto' }}>
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.lg,
            minHeight: '100%',
            flexWrap: 'nowrap',
          }}
        >
          {modelTraces.length === 0 ? (
            <Typography.Text>
              {intl.formatMessage({ defaultMessage: 'Loading traces…', description: 'Loading traces message' })}
            </Typography.Text>
          ) : (
            modelTraces.map((modelTrace, index) => (
              <div
                key={traces[index].evaluationId}
                css={{
                  flex: '1 1 0',
                  minHeight: '100%',
                  minWidth: 0,
                }}
              >
                <ModelTraceExplorer modelTrace={modelTrace} initialActiveView="summary" isInComparisonView />
              </div>
            ))
          )}
        </div>
      </div>
    </Modal>
  );
};
