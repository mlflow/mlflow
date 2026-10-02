import { useState } from 'react';
import { Button, PlayIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';

import type { Skill, SkillStatus } from '../types';
import { getSkillPermissions } from '../utils';
import { UseSkillModal } from './UseSkillModal';

export const UseSkillButton = ({
  skill,
  version,
  versionStatus,
  showLabel = false,
  appearance = 'tertiary',
}: {
  skill: Skill;
  version?: number;
  versionStatus?: SkillStatus;
  showLabel?: boolean;
  appearance?: 'default' | 'tertiary';
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [useModalOpen, setUseModalOpen] = useState(false);
  const { canUse } = getSkillPermissions(skill);
  const pinnedVersion = version ?? skill.latest_version ?? undefined;
  const pinnedStatus = versionStatus ?? (pinnedVersion != null ? (skill.status ?? undefined) : undefined);
  const label = intl.formatMessage({
    defaultMessage: 'Use',
    description: 'Button to use a skill from the Skill Registry catalog',
  });
  const permissionTooltip = intl.formatMessage({
    defaultMessage: 'You do not have permission to use this skill.',
    description: 'Tooltip shown when a user cannot use a skill',
  });

  const useButton = (
    <span onClick={(e) => e.stopPropagation()} css={{ display: 'inline-flex' }}>
      <Button
        componentId="mlflow.skill_registry.use"
        type={appearance === 'tertiary' ? 'tertiary' : undefined}
        size={appearance === 'default' ? 'middle' : 'small'}
        icon={<PlayIcon />}
        aria-label={label}
        disabled={!canUse}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setUseModalOpen(true);
        }}
        css={appearance === 'tertiary' && canUse ? { color: theme.colors.actionPrimaryBackgroundDefault } : undefined}
      >
        {showLabel ? label : undefined}
      </Button>
      {useModalOpen && (
        <UseSkillModal
          visible={useModalOpen}
          skill={skill}
          version={pinnedVersion}
          versionStatus={pinnedStatus}
          onClose={() => setUseModalOpen(false)}
        />
      )}
    </span>
  );

  return canUse ? (
    useButton
  ) : (
    <Tooltip content={permissionTooltip} componentId="mlflow.skill_registry.use.permission_tooltip">
      {useButton}
    </Tooltip>
  );
};
