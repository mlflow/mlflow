import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { useIntl } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import type { RegisteredPromptVersion } from '../types';
import type { PromptsVersionsTableMetadata } from '../utils';
import { ModelVersionTableAliasesCell } from '../../../../model-registry/components/aliases/ModelVersionTableAliasesCell';

export const PromptVersionsTableCombinedCell: ColumnDef<RegisteredPromptVersion>['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { aliasesByVersion, showEditAliasesModal, registeredPrompt } = meta as PromptsVersionsTableMetadata;
  const aliases = aliasesByVersion[original.version] || [];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
        <Typography.Text bold>Version {original.version}</Typography.Text>
        {registeredPrompt && (
          <ModelVersionTableAliasesCell
            modelName={registeredPrompt.name}
            version={original.version}
            aliases={aliases}
            onAddEdit={() => {
              showEditAliasesModal?.(original.version);
            }}
          />
        )}
      </div>
      <Typography.Text size="sm" color="secondary">
        {Utils.formatTimestamp(original.creation_timestamp, intl)}
      </Typography.Text>
    </div>
  );
};
