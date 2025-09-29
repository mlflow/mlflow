import { Button, PencilIcon } from '@databricks/design-system';
import type { ModelEntity } from '../../../experiment-tracking/types';
import { useEditAliasesModal } from '../../../common/hooks/useEditAliasesModal';
import { AliasTag } from '../../../common/components/AliasTag';
import { useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { FormattedMessage } from 'react-intl';
import type { ThunkDispatch } from '../../../redux-types';
import { setModelVersionAliasesApi } from '../../actions';
import { mlflowAliasesLearnMoreLink } from '../../constants';

const getAliasesModalTitle = (version: string) => (
  <FormattedMessage
    defaultMessage="Add/Edit alias for model version {version}"
    description="Model registry > model version alias editor > Title of the update alias modal"
    values={{ version }}
  />
);

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
  const dispatch = useDispatch<ThunkDispatch>();

  const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
    aliases: modelEntity?.aliases ?? [],
    onSuccess: onAliasesModified,
    onSave: async (currentlyEditedVersion: string, existingAliases: string[], draftAliases: string[]) =>
      dispatch(
        setModelVersionAliasesApi(modelEntity?.name ?? '', currentlyEditedVersion, existingAliases, draftAliases),
      ),
    getTitle: getAliasesModalTitle,
    description: (
      <FormattedMessage
        defaultMessage="Aliases allow you to assign a mutable, named reference to a particular model version. <link>Learn more</link>"
        description="Explanation of registered model aliases"
        values={{
          link: (chunks) => (
            <a href={mlflowAliasesLearnMoreLink} rel="noreferrer" target="_blank">
              {chunks}
            </a>
          ),
        }}
      />
    ),
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
            <AliasTag compact value={alias} key={alias} />
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
