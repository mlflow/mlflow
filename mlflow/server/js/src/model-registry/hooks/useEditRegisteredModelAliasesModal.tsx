import { isEqual } from 'lodash';
import { useCallback, useMemo, useState } from 'react';

import { Alert, Button, Form, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { Typography } from '@databricks/design-system';
import { ModelEntity } from '../../experiment-tracking/types';
import { ModelVersionAliasSelect } from '../components/aliases/ModelVersionAliasSelect';
import { FormattedMessage } from 'react-intl';
import { useDispatch } from 'react-redux';
import { ThunkDispatch } from '../../redux-types';
import { setModelVersionAliasesApi } from '../actions';
import { mlflowAliasesLearnMoreLink } from '../constants';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';

const MAX_ALIASES_PER_MODEL_VERSION = 10;

/**
 * Provides methods to initialize and display modal used to add and remove aliases from the model version
 */
export const useEditRegisteredModelAliasesModal = ({
  model,
  onSuccess,
}: {
  model: null | ModelEntity;
  onSuccess?: () => void;
}) => {
  const [showModal, setShowModal] = useState(false);
  const [form] = Form.useForm();

  const [errorMessage, setErrorMessage] = useState<string>('');
  const { theme } = useDesignSystemTheme();

  // We will keep version's existing aliases in `existingAliases` state array
  const [existingAliases, setExistingAliases] = useState<string[]>([]);
  // Currently edited aliases will be kept in `draftAliases` state array
  const [draftAliases, setDraftAliases] = useState<string[]>([]);
  // Currently edited version
  const [currentlyEditedVersion, setCurrentlyEditedVersion] = useState<string>('0');

  const dispatch = useDispatch<ThunkDispatch>();

  /**
   * Function used to invoke the modal and start editing aliases of the particular model version
   */
  const showEditAliasesModal = useCallback(
    (versionNumber: string) => {
      if (!model) {
        return;
      }

      const modelVersionAliases =
        model.aliases?.filter(({ version }) => version === versionNumber).map(({ alias }) => alias) || [];

      if (versionNumber) {
        setExistingAliases(modelVersionAliases);
        setDraftAliases(modelVersionAliases);
        setCurrentlyEditedVersion(versionNumber);
        setShowModal(true);
      }
    },
    [model],
  );

  // // Finds and stores alias values found in other model versions
  const conflictedAliases = useMemo(() => {
    if (!model?.aliases) {
      return [];
    }
    const versionsWithAliases = model.aliases.reduce<{ version: string; aliases: string[] }[]>(
      (aliasMap, aliasEntry) => {
        if (!aliasMap.some(({ version }) => version === aliasEntry.version)) {
          return [...aliasMap, { version: aliasEntry.version, aliases: [aliasEntry.alias] }];
        }
        aliasMap.find(({ version }) => version === aliasEntry.version)?.aliases.push(aliasEntry.alias);
        return aliasMap;
      },
      [],
    );
    const otherVersionMappings = versionsWithAliases.filter(
      ({ version: otherVersion }) => otherVersion !== currentlyEditedVersion,
    );
    return draftAliases
      .map((alias) => ({
        alias,
        otherVersion: otherVersionMappings.find((version) =>
          version.aliases?.find((alias_name) => alias_name === alias),
        ),
      }))
      .filter(({ otherVersion }) => otherVersion);
  }, [model?.aliases, draftAliases, currentlyEditedVersion]);

  // Maps particular aliases to versions
  const aliasToVersionMap = useMemo(
    () =>
      model?.aliases?.reduce<Record<string, string>>((result, { alias, version }) => {
        return { ...result, [alias]: version };
      }, {}) || {},
    [model],
  );

  const save = () => {
    if (!model) {
      return;
    }
    setErrorMessage('');
    dispatch(setModelVersionAliasesApi(model.name, currentlyEditedVersion, existingAliases, draftAliases))
      .then(() => {
        setShowModal(false);
        onSuccess?.();
      })
      .catch((e: ErrorWrapper) => {
        const extractedErrorMessage = e.getMessageField() || e.getUserVisibleError().toString() || e.text;
        setErrorMessage(extractedErrorMessage);
      });
  };

  // Indicates if there is any pending change to the alias set
  const isPristine = isEqual(existingAliases.slice().sort(), draftAliases.slice().sort());
  const isExceedingLimit = draftAliases.length > MAX_ALIASES_PER_MODEL_VERSION;

  const isInvalid = isPristine || isExceedingLimit;

  const EditAliasesModal = (
    <Modal
      visible={showModal}
      footer={
        <div>
          <Button
            componentId="codegen_mlflow_app_src_model-registry_hooks_useeditregisteredmodelaliasesmodal.tsx_131"
            onClick={() => setShowModal(false)}
          >
            <FormattedMessage
              defaultMessage="Cancel"
              description="Model registry > model version alias editor > Cancel editing aliases"
            />
          </Button>
          <Button
            componentId="codegen_mlflow_app_src_model-registry_hooks_useeditregisteredmodelaliasesmodal.tsx_137"
            loading={false}
            type="primary"
            disabled={isInvalid}
            onClick={save}
          >
            <FormattedMessage
              defaultMessage="Save aliases"
              description="Model registry > model version alias editor > Confirm change of aliases"
            />
          </Button>
        </div>
      }
      destroyOnClose
      title={
        <FormattedMessage
          defaultMessage="Add/Edit alias for model version {version}"
          description="Model registry > model version alias editor > Title of the update alias modal"
          values={{ version: currentlyEditedVersion }}
        />
      }
      onCancel={() => setShowModal(false)}
      confirmLoading={false}
    >
      <Typography.Paragraph>
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
      </Typography.Paragraph>
      <Form form={form} layout="vertical">
        <Form.Item>
          <ModelVersionAliasSelect
            disabled={false}
            renderKey={conflictedAliases} // todo
            aliasToVersionMap={aliasToVersionMap}
            version={currentlyEditedVersion}
            draftAliases={draftAliases}
            existingAliases={existingAliases}
            setDraftAliases={setDraftAliases}
          />
        </Form.Item>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {isExceedingLimit && (
            <Alert
              role="alert"
              message={
                <FormattedMessage
                  defaultMessage="You are exceeding a limit of {limit} aliases assigned to the single model version"
                  description="Model registry > model version alias editor > Warning about exceeding aliases limit"
                  values={{ limit: MAX_ALIASES_PER_MODEL_VERSION }}
                />
              }
              type="error"
              closable={false}
            />
          )}
          {conflictedAliases.map(({ alias, otherVersion }) => (
            <Alert
              role="alert"
              key={alias}
              message={
                <FormattedMessage
                  defaultMessage='The "{alias}" alias is also being used on version {otherVersion}. Adding it to this version will remove it from version {otherVersion}.'
                  description="Model registry > model version alias editor > Warning about reusing alias from the other version"
                  values={{ otherVersion: otherVersion?.version, alias }}
                />
              }
              type="info"
              closable={false}
            />
          ))}
          {errorMessage && <Alert role="alert" message={errorMessage} type="error" closable={false} />}
        </div>
      </Form>
    </Modal>
  );

  return { EditAliasesModal, showEditAliasesModal };
};
