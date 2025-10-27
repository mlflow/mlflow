import { useState } from 'react';

import { useDesignSystemTheme, Typography, SparkleIcon, UserIcon, CodeIcon } from '@databricks/design-system';

import { AssessmentActionsOverflowMenu } from './AssessmentActionsOverflowMenu';
import { AssessmentDeleteModal } from './AssessmentDeleteModal';
import { AssessmentSourceName } from './AssessmentSourceName';
import { timeSinceStr } from './AssessmentsPane.utils';
import type { Assessment } from '../ModelTrace.types';
import { getSourceIcon } from './utils';

export const AssessmentItemHeader = ({
  // connector is not displayed in history items
  renderConnector = true,
  assessment,
  setIsEditing,
}: {
  renderConnector?: boolean;
  assessment: Assessment;
  setIsEditing?: (isEditing: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const SourceIcon = getSourceIcon(assessment.source);

  return (
    <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
      {renderConnector && (
        <div
          css={{
            position: 'absolute',
            left: -1,
            top: 0,
            width: 10,
            height: theme.typography.lineHeightBase,
            boxSizing: 'border-box',
            borderBottomLeftRadius: theme.borders.borderRadiusMd,
            borderBottom: `1px solid ${theme.colors.border}`,
            borderLeft: `1px solid ${theme.colors.border}`,
          }}
        />
      )}
      <SourceIcon
        size={theme.typography.fontSizeSm}
        css={{
          padding: 2,
          backgroundColor: theme.colors.actionIconBackgroundHover,
          borderRadius: theme.borders.borderRadiusFull,
        }}
      />
      <AssessmentSourceName source={assessment.source} />
      <div
        css={{
          marginLeft: 'auto',
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing.xs,
        }}
      >
        {assessment.last_update_time && (
          <Typography.Text
            color="secondary"
            size="sm"
            css={{
              marginLeft: theme.spacing.sm,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              textWrap: 'nowrap',
            }}
          >
            {timeSinceStr(new Date(assessment.last_update_time))}
          </Typography.Text>
        )}
        <AssessmentActionsOverflowMenu
          assessment={assessment}
          setIsEditing={setIsEditing}
          setShowDeleteModal={setShowDeleteModal}
        />
        <AssessmentDeleteModal
          assessment={assessment}
          isModalVisible={showDeleteModal}
          setIsModalVisible={setShowDeleteModal}
        />
      </div>
    </div>
  );
};
