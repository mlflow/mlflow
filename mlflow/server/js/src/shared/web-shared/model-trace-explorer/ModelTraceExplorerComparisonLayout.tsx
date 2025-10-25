import { useMemo, useCallback, type ReactNode } from 'react';
import type React from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { AssessmentsPane } from './assessments-pane/AssessmentsPane';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';

export const ModelTraceExplorerComparisonLayout = ({
  header,
  children,
  onScrollContainerRef,
  onScroll,
}: {
  header: ReactNode;
  children: ReactNode;
  onScrollContainerRef?: (element: HTMLDivElement | null) => void;
  onScroll?: (event: React.UIEvent<HTMLDivElement>) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { rootNode, nodeMap, assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled } =
    useModelTraceExplorerViewState();

  const allAssessments = useMemo(() => Object.values(nodeMap).flatMap((node) => node.assessments), [nodeMap]);

  const shouldRenderAssessments = assessmentsPaneEnabled && rootNode;

  const toggleAssessmentsPane = () => {
    setAssessmentsPaneExpanded?.(!assessmentsPaneExpanded);
  };

  const handleScrollContainerRef = useCallback(
    (element: HTMLDivElement | null) => {
      onScrollContainerRef?.(element);
    },
    [onScrollContainerRef],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div
        data-testid="model-trace-explorer-comparison-scroll-container"
        data-comparison-scroll-container
        ref={handleScrollContainerRef}
        onScroll={onScroll}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
        }}
      >
        {header}
        {shouldRenderAssessments && (
          <div css={{ padding: `0 ${theme.spacing.md}`, flexShrink: 0, marginBottom: theme.spacing.sm }}>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                paddingBottom: theme.spacing.xs,
              }}
            >
              <Button
                componentId="shared.model-trace-explorer.comparison.toggle-assessments"
                type="tertiary"
                size="small"
                icon={assessmentsPaneExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                onClick={toggleAssessmentsPane}
              />
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Assessments"
                  description="Header label for the assessments section in the comparison view"
                />
              </Typography.Text>
            </div>
            {assessmentsPaneExpanded && rootNode && (
              <div
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  overflowY: 'auto',
                }}
              >
                <AssessmentsPane assessments={allAssessments} traceId={rootNode.traceId} activeSpanId={undefined} />
              </div>
            )}
          </div>
        )}
        {children}
      </div>
    </div>
  );
};
