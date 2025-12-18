export interface SecretFormData {
  name: string;
  /** Selected authentication mode */
  authMode: string;
  /** Secret field values (e.g., api_key, aws_access_key_id, aws_secret_access_key) */
  secretFields: Record<string, string>;
  /** Additional config field values (e.g., aws_region_name) */
  configFields: Record<string, string>;
}

export const DEFAULT_SECRET_FORM_DATA: SecretFormData = {
  name: '',
  authMode: '',
  secretFields: {},
  configFields: {},
};
