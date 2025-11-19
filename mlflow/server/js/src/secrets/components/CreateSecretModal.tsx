import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  QuestionMarkIcon,
  Input,
  LegacyTooltip,
  Modal,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCreateSecretMutation } from '../hooks/useCreateSecretMutation';
import { useCallback, useState, useMemo } from 'react';

type SecretScope = 'global' | 'scorer';

export const CreateSecretModal = ({ visible, onCancel }: { visible: boolean; onCancel: () => void }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [scope, setScope] = useState<SecretScope>('global');
  const [secretName, setSecretName] = useState('');
  const [envVarKey, setEnvVarKey] = useState('');
  const [secretValue, setSecretValue] = useState('');
  const [errors, setErrors] = useState<{
    secretName?: string;
    envVarKey?: string;
    secretValue?: string;
    scope?: string;
  }>({});

  const isGlobalScope = useMemo(() => scope === 'global', [scope]);
  const secretNameRequired = isGlobalScope;

  const { createSecret, isLoading } = useCreateSecretMutation({
    onSuccess: () => {
      // Reset form and close modal when secret is created
      setScope('global');
      setSecretName('');
      setEnvVarKey('');
      setSecretValue('');
      setErrors({});
      onCancel();
    },
    onError: (error: Error) => {
      const errorMessage = error.message || '';

      // Check if the error is due to FileStore not supporting secrets
      if (errorMessage.includes('FileStore') || errorMessage.includes('NotImplementedError')) {
        setErrors({
          secretValue: intl.formatMessage({
            defaultMessage:
              'Secrets are only supported with database backends (SQLite, PostgreSQL, MySQL). The current backend is using file storage.',
            description: 'Secrets not supported with FileStore error message',
          }),
        });
      } else if (errorMessage.includes('MLFLOW_SECRETS_KEK_PASSPHRASE')) {
        setErrors({
          secretValue: intl.formatMessage({
            defaultMessage:
              'Secrets storage is not configured on the tracking server. Please contact an administrator to enable secrets management.',
            description: 'Secrets KEK passphrase not configured error message',
          }),
        });
      } else if (errorMessage.includes('already exists')) {
        setErrors({
          secretName: intl.formatMessage({
            defaultMessage: 'A secret with this name already exists. Please choose a different name.',
            description: 'Secret name conflict error message',
          }),
        });
      } else {
        setErrors({
          secretValue: intl.formatMessage({
            defaultMessage: 'Failed to create secret. Please try again.',
            description: 'Secret creation failed error message',
          }),
        });
      }
    },
  });

  const handleCreateSecret = useCallback(() => {
    const newErrors: typeof errors = {};

    if (secretNameRequired && !secretName.trim()) {
      newErrors.secretName = intl.formatMessage({
        defaultMessage: 'Secret name is required for global secrets',
        description: 'Secret name required validation message',
      });
    }

    if (!envVarKey.trim()) {
      newErrors.envVarKey = intl.formatMessage({
        defaultMessage: 'Environment variable key is required',
        description: 'Environment variable key required validation message',
      });
    }

    if (!secretValue) {
      newErrors.secretValue = intl.formatMessage({
        defaultMessage: 'Secret credential is required',
        description: 'Secret credential required validation message',
      });
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    const finalSecretName = secretName.trim() || crypto.randomUUID();

    if (isGlobalScope) {
      createSecret({
        secret_name: finalSecretName,
        secret_value: secretValue,
        is_shared: true,
      });
    } else {
      setErrors({
        scope: intl.formatMessage({
          defaultMessage: 'Resource-scoped secrets are not yet supported',
          description: 'Resource scope not supported error message',
        }),
      });
    }
  }, [createSecret, secretName, envVarKey, secretValue, secretNameRequired, isGlobalScope, intl]);

  const handleCancel = useCallback(() => {
    // Reset form when canceling
    setScope('global');
    setSecretName('');
    setEnvVarKey('');
    setSecretValue('');
    setErrors({});
    onCancel();
  }, [onCancel]);

  const isFormValid = useMemo(() => {
    const hasSecretName = secretNameRequired ? secretName.trim().length > 0 : true;
    const hasEnvVarKey = envVarKey.trim().length > 0;
    const hasSecretValue = secretValue.length > 0;
    return hasSecretName && hasEnvVarKey && hasSecretValue;
  }, [secretName, envVarKey, secretValue, secretNameRequired]);

  return (
    <Modal
      componentId="mlflow.secrets.create_secret_modal"
      visible={visible}
      onCancel={handleCancel}
      okText={intl.formatMessage({ defaultMessage: 'Create', description: 'Create secret button text' })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel create secret button text',
      })}
      onOk={handleCreateSecret}
      okButtonProps={{ loading: isLoading, disabled: !isFormValid }}
      title={<FormattedMessage defaultMessage="Create Secret" description="Create secret modal title" />}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <DialogCombobox
            componentId="mlflow.secrets.create_secret_modal.scope"
            label={intl.formatMessage({
              defaultMessage: 'Scope',
              description: 'Secret scope label',
            })}
            value={[scope === 'global' ? 'Global' : 'Scorer']}
          >
            <DialogComboboxTrigger allowClear={false} renderDisplayedValue={(value) => value} />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSelectItem
                  checked={scope === 'global'}
                  value="Global"
                  onChange={() => {
                    setScope('global');
                    setErrors((prev) => ({ ...prev, scope: undefined }));
                  }}
                >
                  <FormattedMessage defaultMessage="Global" description="Secret scope global option" />
                </DialogComboboxOptionListSelectItem>
                {/* TODO: Enable this option when implementing resource-scoped secrets for Scorers.
                     To enable: (1) remove the 'disabled' prop below, (2) update handleCreateSecret to support
                     SCORER_JOB resource_type with proper resource_id selection, (3) add resource picker UI */}
                <DialogComboboxOptionListSelectItem
                  checked={scope === 'scorer'}
                  value="Scorer"
                  onChange={() => {
                    setScope('scorer');
                    setErrors((prev) => ({ ...prev, scope: undefined }));
                  }}
                  disabled
                >
                  <FormattedMessage defaultMessage="Scorer (Coming Soon)" description="Secret scope scorer option" />
                </DialogComboboxOptionListSelectItem>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
          {errors.scope && <FormUI.Message type="error" message={errors.scope} />}
        </div>

        <div>
          <FormUI.Label htmlFor="secret-name-input">
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Secret Name" description="Secret name label" />
              {!secretNameRequired && (
                <span css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
                  (optional)
                </span>
              )}
              <LegacyTooltip
                title={
                  secretNameRequired
                    ? intl.formatMessage({
                        defaultMessage:
                          'A friendly nickname to identify this secret (e.g., "anthropic-general-use-dev", "openai-prod-key"). Required for global secrets.',
                        description: 'Secret name tooltip for global secrets',
                      })
                    : intl.formatMessage({
                        defaultMessage:
                          'A friendly nickname to identify this secret. Optional for resource-scoped secrets - a UUID will be auto-generated if not provided.',
                        description: 'Secret name tooltip for scoped secrets',
                      })
                }
              >
                <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
              </LegacyTooltip>
            </div>
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.create_secret_modal.name"
            id="secret-name-input"
            autoComplete="off"
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., anthropic-general-use-dev',
              description: 'Secret name placeholder',
            })}
            value={secretName}
            onChange={(e) => {
              setSecretName(e.target.value);
              setErrors((prev) => ({ ...prev, secretName: undefined }));
            }}
          />
          {errors.secretName && <FormUI.Message type="error" message={errors.secretName} />}
        </div>

        <div>
          <FormUI.Label htmlFor="env-var-key-input">
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Environment Variable Key"
                description="Environment variable key label"
              />
              <LegacyTooltip
                title={intl.formatMessage({
                  defaultMessage:
                    'The environment variable name that will be used to inject this secret (e.g., "ANTHROPIC_API_KEY", "OPENAI_API_KEY"). This is how your code will access the secret value.',
                  description: 'Environment variable key tooltip',
                })}
              >
                <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
              </LegacyTooltip>
            </div>
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.create_secret_modal.env_var_key"
            id="env-var-key-input"
            autoComplete="off"
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., ANTHROPIC_API_KEY',
              description: 'Environment variable key placeholder',
            })}
            value={envVarKey}
            onChange={(e) => {
              setEnvVarKey(e.target.value);
              setErrors((prev) => ({ ...prev, envVarKey: undefined }));
            }}
          />
          {errors.envVarKey && <FormUI.Message type="error" message={errors.envVarKey} />}
        </div>

        <div>
          <FormUI.Label htmlFor="secret-value-input">
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Secret Credential" description="Secret credential label" />
              <LegacyTooltip
                title={intl.formatMessage({
                  defaultMessage:
                    'The actual credential value (e.g., API key, password, token). This will be encrypted using envelope encryption and stored securely. Only a masked preview will be visible after creation.',
                  description: 'Secret credential tooltip',
                })}
              >
                <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
              </LegacyTooltip>
            </div>
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.create_secret_modal.value"
            id="secret-value-input"
            type="password"
            autoComplete="off"
            data-form-type="other"
            data-lpignore="true"
            data-1p-ignore="true"
            data-bwignore="true"
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., sk-1234abcd...',
              description: 'Secret credential placeholder',
            })}
            value={secretValue}
            onChange={(e) => {
              setSecretValue(e.target.value);
              setErrors((prev) => ({ ...prev, secretValue: undefined }));
            }}
          />
          {errors.secretValue && <FormUI.Message type="error" message={errors.secretValue} />}
        </div>
      </div>
    </Modal>
  );
};
