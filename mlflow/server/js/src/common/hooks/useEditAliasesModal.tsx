import { isEqual } from 'lodash';
import { useCallback, useMemo, useState } from 'react';

import { Alert, Button, LegacyForm, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { Typography } from '@databricks/design-system';
import { AliasSelect } from '../components/AliasSelect';
import { FormattedMessage } from 'react-intl';

import { ErrorWrapper } from '../utils/ErrorWrapper';
import type { AliasMap } from '../types';

const MAX_ALIASES_PER_MODEL_VERSION = 10;

/**
 * Provides methods to initialize and display modal used to add and remove aliases from the model version
 */
export const useEditAliasesModal = ({
  aliases,
  onSuccess,
  onSave,
  getTitle,
  description,
}: {
  aliases: AliasMap;
  onSuccess?: () => void;
  onSave: (currentlyEditedVersion: string, existingAliases: string[], draftAliases: string[]) => Promise<any>;
  getTitle: (version: string) => React.ReactNode;
  description?: React.ReactNode;
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [form] = LegacyForm.useForm();

  const [errorMessage, setErrorMessage] = useState<string>('');
  const { theme } = useDesignSystemTheme();

  // We will keep version's existing aliases in `existingAliases` state array
  const [existingAliases, setExistingAliases] = useState<string[]>([]);
  // Currently edited aliases will be kept in `draftAliases` state array
  const [draftAliases, setDraftAliases] = useState<string[]>([]);
  // Currently edited version
  const [currentlyEditedVersion, setCurrentlyEditedVersion] = useState<string>('0');

  /**
   * Function used to invoke the modal and start editing aliases of the particular model version
   */
  const showEditAliasesModal = useCallback(
    (versionNumber: string) => {
      const modelVersionAliases =
        aliases.filter(({ version }) => version === versionNumber).map(({ alias }) => alias) || [];

      if (versionNumber) {
        setExistingAliases(modelVersionAliases);
        setDraftAliases(modelVersionAliases);
        setCurrentlyEditedVersion(versionNumber);
        setShowModal(true);
      }
    },
    [aliases],
  );

  // // Finds and stores alias values found in other model versions
  const conflictedAliases = useMemo(() => {
    const versionsWithAliases = aliases.reduce<{ version: string; aliases: string[] }[]>((aliasMap, aliasEntry) => {
      if (!aliasMap.some(({ version }) => version === aliasEntry.version)) {
        return [...aliasMap, { version: aliasEntry.version, aliases: [aliasEntry.alias] }];
      }
      aliasMap.find(({ version }) => version === aliasEntry.version)?.aliases.push(aliasEntry.alias);
      return aliasMap;
    }, []);
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
  }, [aliases, draftAliases, currentlyEditedVersion]);

  // Maps particular aliases to versions
  const aliasToVersionMap = useMemo(
    () =>
      aliases.reduce<Record<string, string>>((result, { alias, version }) => {
        return { ...result, [alias]: version };
      }, {}) || {},
    [aliases],
  );

  const save = () => {
    setErrorMessage('');
    setIsLoading(true);
    onSave(currentlyEditedVersion, existingAliases, draftAliases)
      .then(() => {
        setIsLoading(false);
        setShowModal(false);
        onSuccess?.();
      })
      .catch((e: ErrorWrapper | Error) => {
        setIsLoading(false);
        if (e instanceof ErrorWrapper) {
          const extractedErrorMessage = e.getMessageField() || e.getUserVisibleError().toString() || e.text;
          setErrorMessage(extractedErrorMessage);
        } else {
          setErrorMessage(e.message);
        }
      });
  };

  // Indicates if there is any pending change to the alias set
  const isPristine = isEqual(existingAliases.slice().sort(), draftAliases.slice().sort());
  const isExceedingLimit = draftAliases.length > MAX_ALIASES_PER_MODEL_VERSION;

  const isInvalid = isPristine || isExceedingLimit;

  const EditAliasesModal = (
    <Modal
      componentId="mlflow.edit-aliases-modal"
      visible={showModal}
      footer={
        <div>
          <Button componentId="mlflow.edit-aliases-modal.cancel-button" onClick={() => setShowModal(false)}>
            <FormattedMessage defaultMessage="Cancel" description="Alias editor > Cancel editing aliases" />
          </Button>
          <Button
            componentId="mlflow.edit-aliases-modal.save-button"
            loading={isLoading}
            type="primary"
            disabled={isInvalid}
            onClick={save}
          >
            <FormattedMessage defaultMessage="Save aliases" description="Alias editor > Confirm change of aliases" />
          </Button>
        </div>
      }
      destroyOnClose
      title={getTitle(currentlyEditedVersion)}
      onCancel={() => setShowModal(false)}
      confirmLoading={false}
    >
      <Typography.Paragraph>{description}</Typography.Paragraph>
      <LegacyForm form={form} layout="vertical">
        <LegacyForm.Item>
          <AliasSelect
            disabled={false}
            renderKey={conflictedAliases} // todo
            aliasToVersionMap={aliasToVersionMap}
            version={currentlyEditedVersion}
            draftAliases={draftAliases}
            existingAliases={existingAliases}
            setDraftAliases={setDraftAliases}
          />
        </LegacyForm.Item>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {isExceedingLimit && (
            <Alert
              componentId="mlflow.edit-aliases-modal.exceeding-limit-alert"
              role="alert"
              message={
                <FormattedMessage
                  defaultMessage="You are exceeding a limit of {limit} aliases assigned to the single model version"
                  description="Alias editor > Warning about exceeding aliases limit"
                  values={{ limit: MAX_ALIASES_PER_MODEL_VERSION }}
                />
              }
              type="error"
              closable={false}
            />
          )}
          {conflictedAliases.map(({ alias, otherVersion }) => (
            <Alert
              componentId="mlflow.edit-aliases-modal.conflicted-alias-alert"
              role="alert"
              key={alias}
              message={
                <FormattedMessage
                  defaultMessage='The "{alias}" alias is also being used on version {otherVersion}. Adding it to this version will remove it from version {otherVersion}.'
                  description="Alias editor > Warning about reusing alias from the other version"
                  values={{ otherVersion: otherVersion?.version, alias }}
                />
              }
              type="info"
              closable={false}
            />
          ))}
          {errorMessage && (
            <Alert
              componentId="mlflow.edit-aliases-modal.error-alert"
              role="alert"
              message={errorMessage}
              type="error"
              closable={false}
            />
          )}
        </div>
      </LegacyForm>
    </Modal>
  );

  return { EditAliasesModal, showEditAliasesModal };
};
