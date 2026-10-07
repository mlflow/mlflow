import { useState } from 'react';
import {
  Modal,
  SegmentedControlButton,
  SegmentedControlGroup,
  SimpleSelect,
  SimpleSelectOption,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { Skill, SkillStatus } from '../types';
import {
  formatSkillIdentity,
  formatSkillStatusLabel,
  formatSkillUri,
  SKILL_INSTALL_TARGETS,
  STATUS_TAG_COLOR,
  type SkillInstallTargetId,
} from '../utils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { formatSkillPullCli, formatSkillPullPython } from '../snippets';
import { CopyableSnippet, type SnippetFormat } from './CopyableSnippet';

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
  const workspace = useActiveWorkspace();
  const [format, setFormat] = useState<SnippetFormat>('cli');
  const [targetId, setTargetId] = useState<SkillInstallTargetId>('claude-code');
  const destination = SKILL_INSTALL_TARGETS.find((target) => target.id === targetId)?.destination ?? './skills';
  const uri = formatSkillUri(skill.name, skill.organization, version);
  const cliSnippet = formatSkillPullCli(uri, destination, workspace);
  const pythonSnippet = formatSkillPullPython({
    name: skill.name,
    organization: skill.organization || undefined,
    version,
    destination,
    workspace,
  });
  const snippet = format === 'cli' ? cliSnippet : pythonSnippet;
  const identity = formatSkillIdentity(skill.name, skill.organization);

  const installLabel = intl.formatMessage({
    defaultMessage: 'Install for',
    description: 'Label for the Skill use-modal agent destination selector',
  });

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
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.md,
            flexWrap: 'wrap',
          }}
        >
          <SegmentedControlGroup
            name="mlflow.skill_registry.use_modal.format"
            componentId="mlflow.skill_registry.use_modal.format"
            value={format}
            onChange={(event) => setFormat(event.target.value as SnippetFormat)}
          >
            <SegmentedControlButton value="cli">
              <FormattedMessage defaultMessage="CLI" description="CLI example format in the use-skill modal" />
            </SegmentedControlButton>
            <SegmentedControlButton value="python">
              <FormattedMessage defaultMessage="Python" description="Python example format in the use-skill modal" />
            </SegmentedControlButton>
          </SegmentedControlGroup>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text>
              <label htmlFor="mlflow.skill_registry.use_modal.install_target">{installLabel}</label>
            </Typography.Text>
            <SimpleSelect
              id="mlflow.skill_registry.use_modal.install_target"
              componentId="mlflow.skill_registry.use_modal.install_target"
              aria-label={installLabel}
              value={targetId}
              onChange={({ target }) => setTargetId(target.value as SkillInstallTargetId)}
              css={{ width: 180, flex: '0 0 auto' }}
            >
              {SKILL_INSTALL_TARGETS.map((target) => (
                <SimpleSelectOption key={target.id} value={target.id}>
                  {target.label}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
        </div>
        <CopyableSnippet
          componentId="mlflow.skill_registry.use_modal.copy"
          code={snippet}
          format={format}
          copyLabel={intl.formatMessage({
            defaultMessage: 'Copy pull command',
            description: 'Aria label for copying the Skill pull command',
          })}
        />
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Fetches the content from its source into {destination}, where the agent looks for skills."
            description="Hint under the Skill pull command explaining the install destination"
            values={{ destination }}
          />
        </Typography.Text>
      </div>
    </Modal>
  );
};
