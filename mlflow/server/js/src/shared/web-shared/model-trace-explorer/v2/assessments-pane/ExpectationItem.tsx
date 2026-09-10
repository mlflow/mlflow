import { useState } from 'react';

import { Typography, useDesignSystemTheme, ChevronRightIcon, ChevronDownIcon, Button } from '@databricks/design-system';

import { AssessmentActionsOverflowMenu } from '../../assessments-pane/AssessmentActionsOverflowMenu';
import { AssessmentDeleteModal } from './AssessmentDeleteModal';
import { AssessmentEditForm } from './AssessmentEditForm';
import { AssessmentSourceName } from '../../assessments-pane/AssessmentSourceName';
import { getParsedExpectationValue } from '../../assessments-pane/AssessmentsPane.utils';
import { ExpectationValuePreview } from '../../assessments-pane/ExpectationValuePreview';
import { getSourceIcon } from '../../assessments-pane/utils';
import type { ExpectationAssessment } from '../ModelTrace.types';

export const ExpectationItem = ({ expectation }: { expectation: ExpectationAssessment }): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
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
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <div css={{ display: 'flex', alignItems: 'center' }}>
            <SourceIcon
              size={theme.typography.fontSizeSm}
              css={{
                padding: 2,
                backgroundColor: theme.colors.actionIconBackgroundHover,
                borderRadius: theme.borders.borderRadiusFull,
              }}
            />
            <AssessmentSourceName source={expectation.source} />
          </div>
          <div css={{ display: 'flex', alignItems: isExpanded ? 'flex-start' : 'center', gap: theme.spacing.xs }}>
            <Button
              componentId="shared.model-trace-explorer.toggle-expectation-expanded"
              size="small"
              icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
              onClick={() => setIsExpanded(!isExpanded)}
            />
            <div css={{ flex: 1, minWidth: 0 }}>
              {isExpanded ? (
                <div
                  css={{
                    backgroundColor: theme.colors.backgroundSecondary,
                    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
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
    </div>
  );
};
