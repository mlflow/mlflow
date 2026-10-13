import { useCallback, useMemo } from 'react';

import { FileDocumentIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentDisplayValue } from '../assessments-pane/AssessmentDisplayValue';
import { getAssessmentDisplayName } from '../assessments-pane/AssessmentsPane.utils';
import { KeyValueTag } from '../key-value-tag/KeyValueTag';

export function ModelTraceExplorerRetrieverDocumentPreview({
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
    <div
      role="button"
      onClick={() => {
        setExpanded(true);
        logDocumentClick?.('expand');
      }}
      css={{
        display: 'flex',
        flexDirection: 'row',
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        gap: theme.spacing.sm,
        alignItems: 'center',
        justifyContent: 'space-between',
        cursor: 'pointer',
        '&:hover': {
          backgroundColor: theme.colors.backgroundSecondary,
        },
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.sm,
          alignItems: 'center',
          minWidth: 0,
          flexShrink: 1,
        }}
      >
        <FileDocumentIcon />
        <Typography.Text ellipsis size="md">
          {text}
        </Typography.Text>
      </div>
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.sm,
          alignItems: 'center',
          flexShrink: 0,
        }}
      >
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
        {metadataTags.length > 0 ? (
          <KeyValueTag css={{ margin: 0 }} itemKey={metadataTags[0].key} itemValue={metadataTags[0].value} />
        ) : null}
        {metadataTags.length > 1 ? (
          <Tooltip
            componentId="shared.model-trace-explorer.tag-count.hover-tooltip"
            content={metadataTags.slice(1).map(({ key, value }) => (
              <span key={key} css={{ display: 'inline-block' }}>
                {`${key}: ${value}`}
              </span>
            ))}
          >
            <Tag componentId="shared.model-trace-explorer.tag-count" css={{ whiteSpace: 'nowrap', margin: 0 }}>
              +{metadataTags.length - 1}
            </Tag>
          </Tooltip>
        ) : null}
      </div>
    </div>
  );
}
