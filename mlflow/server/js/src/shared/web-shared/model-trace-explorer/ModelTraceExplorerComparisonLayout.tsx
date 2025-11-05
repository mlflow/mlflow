import type { ReactNode } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

import { AssessmentsPane } from './assessments-pane/AssessmentsPane';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';

export const ModelTraceExplorerComparisonLayout = ({
  header,
  children,
}: {
  header?: ReactNode;
  children: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const { assessmentsPaneEnabled, rootNode, selectedNode, isInComparisonView } = useModelTraceExplorerViewState();

  if (!isInComparisonView) {
    return (
      <>
        <div css={{ paddingLeft: theme.spacing.md, paddingBottom: theme.spacing.sm }}>{header}</div>
        {children}
      </>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div
        data-testid="model-trace-explorer-comparison-scroll-container"
        data-comparison-scroll-container
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
        }}
      >
        {header && (
          <div css={{ padding: `0 ${theme.spacing.md}`, paddingBottom: theme.spacing.sm, flexShrink: 0 }}>{header}</div>
        )}
        {assessmentsPaneEnabled && (
          <div css={{ padding: `0 ${theme.spacing.md}`, flexShrink: 0, marginBottom: theme.spacing.sm }}>
            <AssessmentsPane
              assessments={rootNode?.assessments ?? []}
              traceId={rootNode?.traceId ?? ''}
              activeSpanId={(selectedNode?.key as string) ?? ''}
            />
          </div>
        )}
        {children}
      </div>
    </div>
  );
};
