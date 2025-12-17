import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { AssessmentsPane } from './assessments-pane/AssessmentsPane';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { ModelTrace } from './ModelTrace.types';
import { ModelTraceExplorerContent } from './ModelTraceExplorerContent';

export const ModelTraceExplorerComparisonView = ({
  modelTraceInfo,
  className,
  selectedSpanId,
  onSelectSpan,
}: {
  modelTraceInfo: ModelTrace['info'];
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { assessmentsPaneEnabled, assessmentsPaneExpanded, setAssessmentsPaneExpanded, rootNode, selectedNode } =
    useModelTraceExplorerViewState();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          padding: `0px ${theme.spacing.sm}px`,
          marginBottom: theme.spacing.xs,
        }}
      >
        <Button
          size="small"
          componentId="shared.model-trace-explorer.toggle-assessments-pane"
          icon={assessmentsPaneExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setAssessmentsPaneExpanded(!assessmentsPaneExpanded)}
        />
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Assessments" description="Label for the assessments pane" />
        </Typography.Text>
      </div>
      {assessmentsPaneEnabled && assessmentsPaneExpanded && (
        <div
          css={{
            marginBottom: theme.spacing.sm,
            padding: theme.spacing.sm,
            paddingTop: 0,
          }}
        >
          <AssessmentsPane
            assessments={rootNode?.assessments ?? []}
            traceId={rootNode?.traceId ?? ''}
            activeSpanId={(selectedNode?.key as string) ?? ''}
          />
        </div>
      )}
      <ModelTraceExplorerContent
        modelTraceInfo={modelTraceInfo}
        className={className}
        selectedSpanId={selectedSpanId}
        onSelectSpan={onSelectSpan}
      />
    </div>
  );
};
