import { useFormContext } from 'react-hook-form';
import { useMemo, useCallback } from 'react';
import { SecretFormFields } from './SecretFormFields';
import type { SecretFormData } from './types';

interface CreateSecretFormProps {
  /** Provider to fetch auth field configuration for */
  provider: string;
  /** Field prefix for react-hook-form nested fields (default: 'newSecret') */
  fieldPrefix?: string;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
}

/**
 * Secret form that integrates with react-hook-form context.
 * Use this when embedding the secret form within a larger form (e.g., CreateEndpointPage).
 *
 * For standalone secret forms (e.g., API key management), use SecretFormFields directly.
 */
export const CreateSecretForm = ({
  provider,
  fieldPrefix = 'newSecret',
  componentIdPrefix = 'mlflow.gateway.create-endpoint',
}: CreateSecretFormProps) => {
  const { watch, setValue, formState } = useFormContext();

  // Watch all secret fields
  const secretData: SecretFormData = {
    name: watch(`${fieldPrefix}.name`) ?? '',
    authMode: watch(`${fieldPrefix}.authMode`) ?? '',
    secretFields: watch(`${fieldPrefix}.secretFields`) ?? {},
    configFields: watch(`${fieldPrefix}.configFields`) ?? {},
  };

  // Extract errors for the secret fields
  const errors = useMemo(() => {
    const getNestedError = (path: string): string | undefined => {
      const parts = path.split('.');
      let current: any = formState.errors;
      for (const part of parts) {
        if (!current) return undefined;
        current = current[part];
      }
      return current?.message as string | undefined;
    };

    const extractFieldErrors = (fieldType: 'secretFields' | 'configFields'): Record<string, string> | undefined => {
      const errorsMap: Record<string, string> = {};
      const fieldPrefixErrors = formState.errors?.[fieldPrefix] as Record<string, any> | undefined;
      const fieldsErrors = fieldPrefixErrors?.[fieldType];
      if (fieldsErrors && typeof fieldsErrors === 'object') {
        for (const [key, error] of Object.entries(fieldsErrors as Record<string, { message?: string }>)) {
          if (error?.message) {
            errorsMap[key] = error.message;
          }
        }
      }
      return Object.keys(errorsMap).length > 0 ? errorsMap : undefined;
    };

    return {
      name: getNestedError(`${fieldPrefix}.name`),
      secretFields: extractFieldErrors('secretFields'),
      configFields: extractFieldErrors('configFields'),
    };
  }, [formState.errors, fieldPrefix]);

  const handleChange = useCallback(
    (newValue: SecretFormData) => {
      setValue(`${fieldPrefix}.name`, newValue.name, { shouldValidate: true });
      setValue(`${fieldPrefix}.authMode`, newValue.authMode, { shouldValidate: true });
      setValue(`${fieldPrefix}.secretFields`, newValue.secretFields, { shouldValidate: true });
      setValue(`${fieldPrefix}.configFields`, newValue.configFields, { shouldValidate: true });
    },
    [setValue, fieldPrefix],
  );

  return (
    <SecretFormFields
      provider={provider}
      value={secretData}
      onChange={handleChange}
      errors={errors}
      componentIdPrefix={componentIdPrefix}
    />
  );
};
