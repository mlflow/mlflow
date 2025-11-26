import { useFormContext } from 'react-hook-form';
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
    value: watch(`${fieldPrefix}.value`) ?? '',
    authConfig: watch(`${fieldPrefix}.authConfig`) ?? {},
  };

  // Extract errors for the secret fields
  const getErrors = () => {
    const getNestedError = (path: string): string | undefined => {
      const parts = path.split('.');
      let current: any = formState.errors;
      for (const part of parts) {
        if (!current) return undefined;
        current = current[part];
      }
      return current?.message as string | undefined;
    };

    const authConfigErrors: Record<string, string> = {};
    const fieldPrefixErrors = formState.errors?.[fieldPrefix] as Record<string, any> | undefined;
    const authConfig = fieldPrefixErrors?.['authConfig'];
    if (authConfig && typeof authConfig === 'object') {
      for (const [key, error] of Object.entries(authConfig as Record<string, { message?: string }>)) {
        if (error?.message) {
          authConfigErrors[key] = error.message;
        }
      }
    }

    return {
      name: getNestedError(`${fieldPrefix}.name`),
      value: getNestedError(`${fieldPrefix}.value`),
      authConfig: Object.keys(authConfigErrors).length > 0 ? authConfigErrors : undefined,
    };
  };

  const handleChange = (newValue: SecretFormData) => {
    setValue(`${fieldPrefix}.name`, newValue.name, { shouldValidate: true });
    setValue(`${fieldPrefix}.value`, newValue.value, { shouldValidate: true });
    setValue(`${fieldPrefix}.authConfig`, newValue.authConfig, { shouldValidate: true });
  };

  return (
    <SecretFormFields
      provider={provider}
      value={secretData}
      onChange={handleChange}
      errors={getErrors()}
      componentIdPrefix={componentIdPrefix}
    />
  );
};
