import { ColumnDef } from '@tanstack/react-table';
import { ModelVersionTableAliasesCell } from '../../../../model-registry/components/aliases/ModelVersionTableAliasesCell';
import { RegisteredPromptVersion } from '../types';
import { PromptsVersionsTableMetadata } from '../utils';

export const PromptVersionsTableAliasesCell: ColumnDef<RegisteredPromptVersion>['cell'] = ({
  getValue,
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { showEditAliasesModal, aliasesByVersion, registeredPrompt } = meta as PromptsVersionsTableMetadata;

  const mvAliases = aliasesByVersion[original.version] || [];

  return registeredPrompt ? (
    <ModelVersionTableAliasesCell
      modelName={registeredPrompt?.name}
      version={original.version}
      aliases={mvAliases}
      onAddEdit={() => {
        showEditAliasesModal?.(original.version);
      }}
    />
  ) : null;
};
