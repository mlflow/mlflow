export interface SecretFormData {
  name: string;
  authMode: string;
  secretFields: Record<string, string>;
  configFields: Record<string, string>;
}

export const DEFAULT_SECRET_FORM_DATA: SecretFormData = {
  name: '',
  authMode: '',
  secretFields: {},
  configFields: {},
};
