import { useState } from 'react';
import {
  SegmentedControlButton,
  SegmentedControlGroup,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { Skill } from '../types';
import { formatSkillUri, SKILL_INSTALL_TARGETS, type SkillInstallTargetId } from '../utils';
import { useTrackingSnippetConnection } from '../../common/utils/trackingSnippetConnection';
import { formatSkillPullCli, formatSkillPullPython } from '../snippets';
import { CopyableSnippet, type SnippetFormat } from './CopyableSnippet';

/**
 * The pull command for a skill, with a CLI/Python switch and the agent to install it for. Shown in the Use dialog
 * and inline on the skill page; `componentId` prefixes the IDs of its controls so the two are logged apart.
 */
export const UseSkillInstructions = ({
  componentId,
  skill,
  version,
}: {
  componentId: string;
  skill: Pick<Skill, 'name' | 'organization'>;
  version?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const connection = useTrackingSnippetConnection();
  const [format, setFormat] = useState<SnippetFormat>('cli');
  const [targetId, setTargetId] = useState<SkillInstallTargetId>('claude-code');
  const skillsDirectory = SKILL_INSTALL_TARGETS.find((target) => target.id === targetId)?.destination ?? './skills';
  // Pull writes the skill's files directly into the destination, so each skill gets its own folder.
  const destination = `${skillsDirectory}/${skill.name}`;
  const uri = formatSkillUri(skill.name, skill.organization, version);
  const snippet =
    format === 'cli'
      ? formatSkillPullCli(uri, destination, connection)
      : formatSkillPullPython(uri, destination, connection);

  const installLabel = intl.formatMessage({
    defaultMessage: 'Install for',
    description: 'Label for the agent selector in the skill pull instructions',
  });

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
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
          name={`${componentId}.format`}
          componentId={`${componentId}.format`}
          value={format}
          onChange={(event) => setFormat(event.target.value as SnippetFormat)}
        >
          <SegmentedControlButton value="cli">
            <FormattedMessage defaultMessage="CLI" description="CLI format of the skill pull instructions" />
          </SegmentedControlButton>
          <SegmentedControlButton value="python">
            <FormattedMessage defaultMessage="Python" description="Python format of the skill pull instructions" />
          </SegmentedControlButton>
        </SegmentedControlGroup>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text>
            <label htmlFor={`${componentId}.install_target`}>{installLabel}</label>
          </Typography.Text>
          <SimpleSelect
            id={`${componentId}.install_target`}
            componentId={`${componentId}.install_target`}
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
        componentId={`${componentId}.copy`}
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
  );
};
