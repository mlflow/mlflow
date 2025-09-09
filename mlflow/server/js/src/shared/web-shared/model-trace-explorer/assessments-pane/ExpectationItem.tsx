import { isObject } from 'lodash';
import { useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { AssessmentActionsOverflowMenu } from './AssessmentActionsOverflowMenu';
import { AssessmentDeleteModal } from './AssessmentDeleteModal';
import { AssessmentEditForm } from './AssessmentEditForm';
import { getParsedExpectationValue } from './AssessmentsPane.utils';
import { ExpectationValuePreview } from './ExpectationValuePreview';
import { SpanNameDetailViewLink } from './SpanNameDetailViewLink';
import type { ExpectationAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

const getValuePreview = (parsedValue: any) => {
  if (Array.isArray(parsedValue)) {
    return parsedValue.map((item, index) => <ExpectationValuePreview value={item} key={index} />);
  }

  if (isObject(parsedValue)) {
    return Object.entries(parsedValue).map(([key, value]) => (
      <ExpectationValuePreview key={key} objectKey={key} value={value} />
    ));
  }

  return <ExpectationValuePreview value={parsedValue} />;
};

export const ExpectationItem = ({ expectation }: { expectation: ExpectationAssessment }) => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const { nodeMap, activeView } = useModelTraceExplorerViewState();

  const associatedSpan = expectation.span_id ? nodeMap[expectation.span_id] : null;
  // the summary view displays all assessments regardless of span, so
  // we need some way to indicate which span an assessment is associated with.
  const showAssociatedSpan = activeView === 'summary' && associatedSpan;

  const parsedValue = getParsedExpectationValue(expectation.expectation);

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
        getValuePreview(parsedValue)
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
