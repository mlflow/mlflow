import { useState } from 'react';
import { Button, PlayIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';

import type { Skill, SkillStatus } from '../types';
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
  const pinnedVersion = version ?? skill.latest_version ?? undefined;
  const pinnedStatus = versionStatus ?? (pinnedVersion != null ? (skill.status ?? undefined) : undefined);
  const label = intl.formatMessage({
    defaultMessage: 'Use',
    description: 'Button to use a skill from the Skill Registry catalog',
  });
  return (
    <span onClick={(e) => e.stopPropagation()} css={{ display: 'inline-flex' }}>
      <Button
        componentId="mlflow.skill_registry.use"
        type={appearance === 'tertiary' ? 'tertiary' : undefined}
        size={appearance === 'default' ? 'middle' : 'small'}
        icon={<PlayIcon />}
        aria-label={label}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setUseModalOpen(true);
        }}
        css={appearance === 'tertiary' ? { color: theme.colors.actionPrimaryBackgroundDefault } : undefined}
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
};
