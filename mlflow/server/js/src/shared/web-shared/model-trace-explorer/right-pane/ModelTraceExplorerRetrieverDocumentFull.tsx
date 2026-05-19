import { useCallback, useMemo } from 'react';

import {
  Button,
  ChevronUpIcon,
  FileDocumentIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentDisplayValue } from '../assessments-pane/AssessmentDisplayValue';
import { getAssessmentDisplayName } from '../assessments-pane/AssessmentsPane.utils';
import { KeyValueTag } from '../key-value-tag/KeyValueTag';

export function ModelTraceExplorerRetrieverDocumentFull({
  text,
  metadataTags,
  setExpanded,
  relevanceAssessment,
  logDocumentClick,
}: {
  text: string;
  metadataTags: { key: string; value: string }[];
  setExpanded: (expanded: boolean) => void;
  relevanceAssessment?: FeedbackAssessment;
  logDocumentClick?: (action: string) => void;
}) {
  const { theme } = useDesignSystemTheme();
  const { highlightAssessment, setAssessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const handleAssessmentBadgeClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (relevanceAssessment) {
        setAssessmentsPaneExpanded(true);
        highlightAssessment(relevanceAssessment.assessment_id);
      }
    },
    [relevanceAssessment, highlightAssessment, setAssessmentsPaneExpanded],
  );

  const assessmentJsonValue = useMemo(
    () => JSON.stringify(relevanceAssessment?.feedback.value) ?? '',
    [relevanceAssessment?.feedback.value],
  );

  const assessmentBadge = relevanceAssessment ? (
    <div
      role="button"
      onClick={handleAssessmentBadgeClick}
      css={{
        cursor: 'pointer',
        '&:hover': { opacity: 0.8 },
      }}
    >
      <AssessmentDisplayValue
        jsonValue={assessmentJsonValue}
        prefix={
          <Typography.Text size="sm" css={{ marginRight: theme.spacing.xs }}>
            {getAssessmentDisplayName(relevanceAssessment.assessment_name)}:
          </Typography.Text>
        }
      />
    </div>
  ) : null;

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div
        role="button"
        onClick={() => {
          setExpanded(false);
          logDocumentClick?.('collapse');
        }}
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          minHeight: theme.typography.lineHeightBase,
          boxSizing: 'content-box',
          '&:hover': {
            backgroundColor: theme.colors.backgroundSecondary,
          },
        }}
      >
        <FileDocumentIcon />
        {relevanceAssessment?.rationale ? (
          <Tooltip
            componentId="shared.model-trace-explorer.relevance-assessment-tooltip"
            content={relevanceAssessment.rationale}
          >
            {assessmentBadge}
          </Tooltip>
        ) : (
          assessmentBadge
        )}
      </div>
      <div css={{ padding: theme.spacing.md, paddingBottom: 0 }}>
        <GenAIMarkdownRenderer>{text}</GenAIMarkdownRenderer>
      </div>
      <div css={{ padding: theme.spacing.md, paddingTop: 0 }}>
        {metadataTags.map(({ key, value }) => (
          <KeyValueTag key={key} itemKey={key} itemValue={value} />
        ))}
      </div>
      <Button
        css={{ width: '100%', padding: theme.spacing.sm }}
        componentId="shared.model-trace-explorer.retriever-document-collapse"
        icon={<ChevronUpIcon />}
        type="tertiary"
        onClick={() => setExpanded(false)}
      >
        <FormattedMessage
          defaultMessage="See less"
          description="Model trace explorer > selected span > code snippet > see less button"
        />
      </Button>
    </div>
  );
}
