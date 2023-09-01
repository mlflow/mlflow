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
        <Button size='small' type='link' onClick={onAddEdit}>
          Add
        </Button>
      ) : (
        <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center' }}>
          {aliases.map((alias) => (
            <ModelVersionAliasTag value={alias} key={alias} />
          ))}
          <Button size='small' icon={<PencilIcon />} onClick={onAddEdit} />
        </div>
      )}
    </>
  );
};
