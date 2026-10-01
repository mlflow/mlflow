import { useState } from 'react';
import { Button, PlayIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';

import type { Skill } from '../types';
import { getSkillPermissions } from '../utils';
import { UseSkillModal } from './UseSkillModal';

export const UseSkillButton = ({ skill, showLabel = false }: { skill: Skill; showLabel?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [useModalOpen, setUseModalOpen] = useState(false);
  const { canUse } = getSkillPermissions(skill);
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
        type="tertiary"
        size="small"
        icon={<PlayIcon />}
        aria-label={label}
        disabled={!canUse}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setUseModalOpen(true);
        }}
        css={canUse ? { color: theme.colors.actionPrimaryBackgroundDefault } : undefined}
      >
        {showLabel ? label : undefined}
      </Button>
      {useModalOpen && <UseSkillModal visible={useModalOpen} skill={skill} onClose={() => setUseModalOpen(false)} />}
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
