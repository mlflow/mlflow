import { useState } from 'react';
import {
  Button,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  formatSkillImportCli,
  formatSkillImportPython,
  formatSkillRegisterCli,
  formatSkillRegisterPython,
  type SkillImportSnippetOptions,
  type SkillRegisterSnippetOptions,
} from '../snippets';
import { CopyableSnippet, type SnippetFormat } from './CopyableSnippet';

export const RepositoryImportHint = ({ code, format }: { code: string; format: SnippetFormat }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text size="sm" color="secondary">
        <FormattedMessage
          defaultMessage="Registering every skill in this repository? Run this instead:"
          description="Pointer from single-skill registration to the repository import command"
        />
      </Typography.Text>
      <CopyableSnippet
        componentId="mlflow.skill_registry.register_modal.repository_import.copy"
        code={code}
        format={format}
        copyLabel={intl.formatMessage({
          defaultMessage: 'Copy repository import command',
          description: 'Aria label for copying the skill repository import command',
        })}
      />
    </div>
  );
};

export const RegisterSkillApiView = ({
  register,
  repositoryImport,
  onBack,
}: {
  register: SkillRegisterSnippetOptions;
  repositoryImport?: SkillImportSnippetOptions;
  onBack: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [format, setFormat] = useState<SnippetFormat>('cli');

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <Button componentId="mlflow.skill_registry.register_modal.back" type="link" onClick={onBack}>
          <FormattedMessage
            defaultMessage="← Back to form"
            description="Return from the skill API example to the form"
          />
        </Button>
      </div>
      <SegmentedControlGroup
        name="mlflow.skill_registry.register_modal.api_format"
        componentId="mlflow.skill_registry.register_modal.api_format"
        value={format}
        onChange={(event) => setFormat(event.target.value as SnippetFormat)}
      >
        <SegmentedControlButton value="cli">
          <FormattedMessage defaultMessage="CLI" description="CLI example for skill registration" />
        </SegmentedControlButton>
        <SegmentedControlButton value="python">
          <FormattedMessage defaultMessage="Python" description="Python example for skill registration" />
        </SegmentedControlButton>
      </SegmentedControlGroup>
      <Typography.Text color="secondary">
        {format === 'cli' ? (
          <FormattedMessage
            defaultMessage="The CLI reads the skill locally, so it can infer the name and record a digest."
            description="Explanation of the skill registration CLI example"
          />
        ) : (
          <FormattedMessage
            defaultMessage="The Python SDK reads the skill locally, so it can infer the name and record a digest."
            description="Explanation of the skill registration Python example"
          />
        )}
      </Typography.Text>
      <CopyableSnippet
        componentId="mlflow.skill_registry.register_modal.api_snippet.copy"
        code={format === 'cli' ? formatSkillRegisterCli(register) : formatSkillRegisterPython(register)}
        format={format}
        copyLabel={intl.formatMessage({
          defaultMessage: 'Copy register command',
          description: 'Aria label for copying the skill registration API example',
        })}
      />
      {repositoryImport && (
        <RepositoryImportHint
          format={format}
          code={format === 'cli' ? formatSkillImportCli(repositoryImport) : formatSkillImportPython(repositoryImport)}
        />
      )}
    </div>
  );
};
