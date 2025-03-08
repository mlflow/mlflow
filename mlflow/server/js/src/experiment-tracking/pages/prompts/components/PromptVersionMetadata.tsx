import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ModelVersionTableAliasesCell } from '../../../../model-registry/components/aliases/ModelVersionTableAliasesCell';
import { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import Utils from '../../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';

export const PromptVersionMetadata = ({
  registeredPromptVersion,
  registeredPrompt,
  showEditAliasesModal,
  onEditVersion,
  aliasesByVersion,
}: {
  registeredPrompt?: RegisteredPrompt;
  registeredPromptVersion?: RegisteredPromptVersion;
  showEditAliasesModal?: (versionNumber: string) => void;
  onEditVersion?: (vesrion: RegisteredPromptVersion) => void;
  aliasesByVersion: Record<string, string[]>;
}) => {
  const { theme } = useDesignSystemTheme();
  if (!registeredPrompt || !registeredPromptVersion) {
    return null;
  }

  const versionElement = (
    <FormattedMessage defaultMessage="Version {version}" values={{ version: registeredPromptVersion.version }} />
  );

  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr',
        gridAutoRows: `minmax(${theme.typography.lineHeightLg}, auto)`,
        alignItems: 'flex-start',
        rowGap: theme.spacing.xs,
        columnGap: theme.spacing.sm,
      }}
    >
      <Typography.Text bold>Version:</Typography.Text>
      <Typography.Text>
        {onEditVersion ? (
          <Typography.Link componentId="TODO" onClick={() => onEditVersion(registeredPromptVersion)}>
            {versionElement}
          </Typography.Link>
        ) : (
          <Typography.Text>{versionElement}</Typography.Text>
        )}{' '}
        (baseline)
      </Typography.Text>
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Registered at:" description="TODO" />
      </Typography.Text>
      <Typography.Text>{Utils.formatTimestamp(registeredPromptVersion.creation_timestamp)}</Typography.Text>
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Aliases:" description="TODO" />
      </Typography.Text>
      <div>
        <ModelVersionTableAliasesCell
          css={{ maxWidth: 'none' }}
          modelName={registeredPrompt.name}
          version={registeredPromptVersion.version}
          aliases={aliasesByVersion[registeredPromptVersion.version] || []}
          onAddEdit={() => {
            showEditAliasesModal?.(registeredPromptVersion.version);
          }}
        />
      </div>
    </div>
  );
};
