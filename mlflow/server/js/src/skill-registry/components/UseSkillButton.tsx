import { useState } from 'react';
import { Button, PlayIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';

import type { Skill } from '../types';
import { UseSkillModal } from './UseSkillModal';

export const UseSkillButton = ({ skill, showLabel = false }: { skill: Skill; showLabel?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [useModalOpen, setUseModalOpen] = useState(false);
  const label = intl.formatMessage({
    defaultMessage: 'Use',
    description: 'Button to use a skill from the Skill Registry catalog',
  });

  return (
    <span onClick={(e) => e.stopPropagation()} css={{ display: 'inline-flex' }}>
      <Button
        componentId="mlflow.skill_registry.use"
        type="tertiary"
        size="small"
        icon={<PlayIcon />}
        aria-label={label}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setUseModalOpen(true);
        }}
        css={{ color: theme.colors.actionPrimaryBackgroundDefault }}
      >
        {showLabel ? label : undefined}
      </Button>
      {useModalOpen && <UseSkillModal visible={useModalOpen} skill={skill} onClose={() => setUseModalOpen(false)} />}
    </span>
  );
};
