/**
 * Form data structure for creating or editing a secret
 */
export interface SecretFormData {
  name: string;
  value: string;
  authConfig: Record<string, string>;
}

/**
 * Initial/default values for a new secret form
 */
export const DEFAULT_SECRET_FORM_DATA: SecretFormData = {
  name: '',
  value: '',
  authConfig: {},
};
