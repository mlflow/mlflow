import { useState } from 'react';
import { Typography, useDesignSystemTheme, ChevronRightIcon, ChevronDownIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { AssessmentActionsOverflowMenu } from './AssessmentActionsOverflowMenu';
import { AssessmentDeleteModal } from './AssessmentDeleteModal';
import { AssessmentEditForm } from './AssessmentEditForm';
import { getParsedExpectationValue } from './AssessmentsPane.utils';
import { ExpectationValuePreview } from './ExpectationValuePreview';
import { SpanNameDetailViewLink } from './SpanNameDetailViewLink';
import type { ExpectationAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { getSourceIcon } from './utils';

export const ExpectationItem = ({ expectation }: { expectation: ExpectationAssessment }) => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const { nodeMap, activeView } = useModelTraceExplorerViewState();

  const associatedSpan = expectation.span_id ? nodeMap[expectation.span_id] : null;
  // the summary view displays all assessments regardless of span, so
  // we need some way to indicate which span an assessment is associated with.
  const showAssociatedSpan = activeView === 'summary' && associatedSpan;

  const parsedValue = getParsedExpectationValue(expectation.expectation);
  const SourceIcon = getSourceIcon(expectation.source);

  return (
    <div
      css={{
        padding: theme.spacing.sm + theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {expectation.assessment_name}
        </Typography.Text>
        <AssessmentActionsOverflowMenu
          assessment={expectation}
          setIsEditing={setIsEditing}
          setShowDeleteModal={setShowDeleteModal}
        />
        <AssessmentDeleteModal
          assessment={expectation}
          isModalVisible={showDeleteModal}
          setIsModalVisible={setShowDeleteModal}
        />
      </div>
      {isEditing ? (
        <AssessmentEditForm
          assessment={expectation}
          onSuccess={() => setIsEditing(false)}
          onCancel={() => setIsEditing(false)}
        />
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <SourceIcon/> 
              <Typography.Text size="sm" color="secondary">
                {expectation.source.source_id}
              </Typography.Text>
            </div>
          <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.xs }}>
            <div
              css={{ paddingTop: 2, flexShrink: 0, cursor: 'pointer' }}
              onClick={(e) => {
                e.stopPropagation();
                setIsExpanded(!isExpanded);
              }}
            >
              {isExpanded ? (
                <ChevronDownIcon css={{ fontSize: 16 }} />
              ) : (
                <ChevronRightIcon css={{ fontSize: 16 }} />
              )}
            </div>
            <div css={{ flex: 1, minWidth: 0 }}>
              {isExpanded ? (
                <div
                  css={{
                    backgroundColor: theme.colors.backgroundSecondary,
                    padding: theme.spacing.sm,
                    borderRadius: theme.borders.borderRadiusMd,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  <Typography.Text>
                    {typeof parsedValue === 'string' ? parsedValue : JSON.stringify(parsedValue, null, 2)}
                  </Typography.Text>
                </div>
              ) : (
                <ExpectationValuePreview parsedValue={parsedValue} singleLine />
              )}
            </div>
          </div>
        </div>
      )}
      {showAssociatedSpan && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage defaultMessage="Span" description="Label for the associated span of an assessment" />
          </Typography.Text>
          <SpanNameDetailViewLink node={associatedSpan} />
        </div>
      )}
    </div>
  );
};
