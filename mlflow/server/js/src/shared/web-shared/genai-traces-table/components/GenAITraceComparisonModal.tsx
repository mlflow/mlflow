import { compact } from 'lodash';
import { useMemo } from 'react';

import { Drawer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import {
  CompareModelTraceExplorer,
  ModelTraceExplorerSkeleton,
  useGetTracesById,
} from '@databricks/web-shared/model-trace-explorer';
import { AssistantAwareDrawer } from '@mlflow/mlflow/src/common/components/AssistantAwareDrawer';

// prettier-ignore
export const GenAITraceComparisonModal = ({
  traceIds,
  onClose,
}: {
  traceIds: string[];
  onClose?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const queryParams = undefined;
  const { data: fetchedTraces, isLoading } = useGetTracesById(traceIds, queryParams);

  const modelTraces = useMemo(() => compact(fetchedTraces), [fetchedTraces]);

  return (
    <AssistantAwareDrawer.Root
      open
      onOpenChange={(open) => {
        if (!open) {
          onClose?.();
        }
      }}
    >
      <AssistantAwareDrawer.Content
        componentId="mlflow.evaluations_review.modal"
        width="90vw"
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Title level={2} withoutMargins>
              <FormattedMessage
                defaultMessage="Comparing {count} traces"
                description="Comparing traces title"
                values={{ count: traceIds.length }}
              />
            </Typography.Title>
          </div>
        }
        expandContentToFullHeight
        css={[
          {
            // Disable drawer's scroll to allow inner content to handle scrolling
            '&>div': {
              overflow: 'hidden',
            },
            '&>div:first-child': {
              paddingLeft: theme.spacing.md,
            },
          },
        ]}
      >
        {isLoading || !modelTraces || modelTraces.length === 0 ? (
          <ModelTraceExplorerSkeleton />
        ) : (
          <CompareModelTraceExplorer
            modelTraces={modelTraces}
            css={{
              marginLeft: -theme.spacing.lg,
              marginRight: -theme.spacing.lg,
            }}
          />
        )}
      </AssistantAwareDrawer.Content>
    </AssistantAwareDrawer.Root>
  );
};
