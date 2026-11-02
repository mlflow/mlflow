import { Modal, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { Skill, SkillStatus } from '../types';
import { formatSkillIdentity, formatSkillStatusLabel, STATUS_TAG_COLOR } from '../utils';
import { UseSkillInstructions } from './UseSkillInstructions';

export const UseSkillModal = ({
  visible,
  skill,
  version,
  versionStatus,
  onClose,
}: {
  visible: boolean;
  skill: Skill;
  version?: number;
  versionStatus?: SkillStatus;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const identity = formatSkillIdentity(skill.name, skill.organization);

  if (!visible) return null;

  return (
    <Modal
      componentId="mlflow.skill_registry.use_modal"
      title={
        <FormattedMessage
          defaultMessage="Use {identity}"
          description="Title for the Skill Registry use-skill modal"
          values={{ identity }}
        />
      }
      visible={visible}
      onCancel={onClose}
      footer={null}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {version != null && (
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Pinned version: v{version}"
                description="Pinned Skill version shown in the use-skill modal"
                values={{ version }}
              />
            </Typography.Text>
            {versionStatus && (
              <Tag componentId="mlflow.skill_registry.use_modal.version_status" color={STATUS_TAG_COLOR[versionStatus]}>
                {formatSkillStatusLabel(intl, versionStatus)}
              </Tag>
            )}
          </span>
        )}
        <UseSkillInstructions componentId="mlflow.skill_registry.use_modal" skill={skill} version={version} />
      </div>
    </Modal>
  );
};
