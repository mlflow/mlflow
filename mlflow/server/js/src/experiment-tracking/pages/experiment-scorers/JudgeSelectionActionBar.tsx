import React from 'react';
import { useDesignSystemTheme, Button, Typography, TrashIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { COMPONENT_ID_PREFIX } from './constants';
import type { ScheduledScorer } from './types';
import JudgeUseDropdown from './JudgeUseDropdown';

interface JudgeSelectionActionBarProps {
  selectedScorers: ScheduledScorer[];
  experimentId: string;
  onDelete: () => void;
  onClearSelection: () => void;
}

const JudgeSelectionActionBar: React.FC<JudgeSelectionActionBarProps> = ({
  selectedScorers,
  experimentId,
  onDelete,
  onClearSelection,
}) => {
  const { theme } = useDesignSystemTheme();

  if (selectedScorers.length === 0) return null;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        borderRadius: theme.general.borderRadiusBase,
        position: 'sticky',
        top: 0,
        zIndex: 10,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.clear-selection`}
          size="small"
          type="tertiary"
          onClick={onClearSelection}
          css={{ color: theme.colors.white }}
        >
          <Typography.Text bold css={{ color: theme.colors.white }}>
            <FormattedMessage
              defaultMessage="{count} selected"
              description="Number of selected judges in action bar"
              values={{ count: selectedScorers.length }}
            />
          </Typography.Text>
        </Button>
      </div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <JudgeUseDropdown selectedScorers={selectedScorers} experimentId={experimentId} />
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.bulk-delete`}
          size="small"
          type="tertiary"
          icon={<TrashIcon />}
          onClick={onDelete}
          css={{ color: theme.colors.white }}
        >
          <FormattedMessage defaultMessage="Delete" description="Bulk delete selected judges button" />
        </Button>
      </div>
    </div>
  );
};

export default JudgeSelectionActionBar;
