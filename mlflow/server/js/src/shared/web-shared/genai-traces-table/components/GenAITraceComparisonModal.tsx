import { ModelTraceExplorer } from '../../model-trace-explorer';
import { useMemo } from 'react';

import { Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { useIntl } from '@databricks/i18n';
import { useFetchTraces } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets/hooks/useFetchTraces';
import { compact } from 'lodash';

export const GenAITraceComparisonModal = ({ traceIds, onClose }: { traceIds: string[]; onClose: () => void }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const { data: fetchedTraces, isLoading } = useFetchTraces({ traceIds });

  const modelTraces = useMemo(() => compact(fetchedTraces), [fetchedTraces]);

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
      <div
        css={{
          height: '100%',
          overflow: 'auto',
          marginLeft: -theme.spacing.lg,
          marginRight: -theme.spacing.lg,
        }}
      >
        <div
          css={{
            display: 'flex',
            minHeight: '100%',
            flexWrap: 'nowrap',
          }}
        >
          {isLoading || !modelTraces || modelTraces.length === 0 ? (
            <Typography.Text>
              {intl.formatMessage({ defaultMessage: 'Loading traces…', description: 'Loading traces message' })}
            </Typography.Text>
          ) : (
            modelTraces.map((modelTrace, index) => (
              <div
                key={traceIds[index]}
                css={{
                  flex: '1 1 0',
                  minHeight: '100%',
                  minWidth: 0,
                  borderRight: index < modelTraces.length - 1 ? `1px solid ${theme.colors.border}` : 'none',
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
