import { useFormContext } from 'react-hook-form';
import { SecretFormFields } from './SecretFormFields';
import type { SecretFormData } from './types';

interface CreateSecretFormProps {
  provider: string;
  fieldPrefix?: string;
  componentIdPrefix?: string;
}

export const CreateSecretForm = ({
  provider,
  fieldPrefix = 'newSecret',
  componentIdPrefix = 'mlflow.gateway.create-endpoint',
}: CreateSecretFormProps) => {
  const { watch, setValue, formState } = useFormContext();

  const secretData: SecretFormData = {
    name: watch(`${fieldPrefix}.name`) ?? '',
    authMode: watch(`${fieldPrefix}.authMode`) ?? '',
    secretFields: watch(`${fieldPrefix}.secretFields`) ?? {},
    configFields: watch(`${fieldPrefix}.configFields`) ?? {},
  };

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

    const extractFieldErrors = (fieldType: 'secretFields' | 'configFields'): Record<string, string> | undefined => {
      const errors: Record<string, string> = {};
      const fieldPrefixErrors = formState.errors?.[fieldPrefix] as Record<string, any> | undefined;
      const fieldsErrors = fieldPrefixErrors?.[fieldType];
      if (fieldsErrors && typeof fieldsErrors === 'object') {
        for (const [key, error] of Object.entries(fieldsErrors as Record<string, { message?: string }>)) {
          if (error?.message) {
            errors[key] = error.message;
          }
        }
      }
      return Object.keys(errors).length > 0 ? errors : undefined;
    };

    return {
      name: getNestedError(`${fieldPrefix}.name`),
      secretFields: extractFieldErrors('secretFields'),
      configFields: extractFieldErrors('configFields'),
    };
  };

  const handleChange = (newValue: SecretFormData) => {
    setValue(`${fieldPrefix}.name`, newValue.name, { shouldValidate: true });
    setValue(`${fieldPrefix}.authMode`, newValue.authMode, { shouldValidate: true });
    setValue(`${fieldPrefix}.secretFields`, newValue.secretFields, { shouldValidate: true });
    setValue(`${fieldPrefix}.configFields`, newValue.configFields, { shouldValidate: true });
  };

  return (
    <SecretFormFields
      provider={provider}
      value={secretData}
      onChange={handleChange}
      errors={getErrors()}
      componentIdPrefix={componentIdPrefix}
      hideNameField
    />
  );
};
