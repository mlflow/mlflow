import { CopyIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import type { Skill } from '../types';
import { formatSkillIdentity } from '../utils';
import { overlayButtonStyles } from '../styles';

export const UseSkillModal = ({ visible, skill, onClose }: { visible: boolean; skill: Skill; onClose: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const identity = formatSkillIdentity(skill.name, skill.organization);
  const snippet = `mlflow skills pull ${identity}`;

  if (!visible) return null;

  return (
    <Modal
      componentId="mlflow.skill_registry.use_modal"
      title={
        <FormattedMessage
          defaultMessage="Use {name}"
          description="Title for the Skill Registry use-skill modal"
          values={{ name: skill.name }}
        />
      }
      visible={visible}
      onCancel={onClose}
      footer={null}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Pull this skill into your local environment:"
            description="Instruction text for the Skill Registry use-skill modal"
          />
        </Typography.Text>
        <div css={{ position: 'relative' }}>
          <CopyButton
            componentId="mlflow.skill_registry.use_modal.copy"
            showLabel={false}
            copyText={snippet}
            icon={<CopyIcon />}
            css={overlayButtonStyles(theme)}
          />
          <CodeSnippet
            language="text"
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{ padding: theme.spacing.sm, paddingRight: theme.spacing.xl + theme.spacing.sm }}
          >
            {snippet}
          </CodeSnippet>
        </div>
      </div>
    </Modal>
  );
};
