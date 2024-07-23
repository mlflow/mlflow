import { Button, PencilIcon } from '@databricks/design-system';
import type { ModelEntity } from '../../../experiment-tracking/types';
import { useEditRegisteredModelAliasesModal } from '../../hooks/useEditRegisteredModelAliasesModal';
import { ModelVersionAliasTag } from './ModelVersionAliasTag';
import { useCallback } from 'react';

export const ModelVersionViewAliasEditor = ({
  aliases = [],
  modelEntity,
  version,
  onAliasesModified,
}: {
  modelEntity?: ModelEntity;
  aliases?: string[];
  version: string;
  onAliasesModified?: () => void;
}) => {
  const { EditAliasesModal, showEditAliasesModal } = useEditRegisteredModelAliasesModal({
    model: modelEntity || null,
    onSuccess: onAliasesModified,
  });
  const onAddEdit = useCallback(() => {
    showEditAliasesModal(version);
  }, [showEditAliasesModal, version]);
  return (
    <>
      {EditAliasesModal}
      {aliases.length < 1 ? (
        <Button
          componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelversionviewaliaseditor.tsx_29"
          size="small"
          type="link"
          onClick={onAddEdit}
          title="Add aliases"
        >
          Add
        </Button>
      ) : (
        <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center' }}>
          {aliases.map((alias) => (
            <ModelVersionAliasTag compact value={alias} key={alias} />
          ))}
          <Button
            componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelversionviewaliaseditor.tsx_37"
            size="small"
            icon={<PencilIcon />}
            onClick={onAddEdit}
            title="Edit aliases"
          />
        </div>
      )}
    </>
  );
};
