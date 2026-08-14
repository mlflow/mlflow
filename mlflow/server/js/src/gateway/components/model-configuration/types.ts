export type SecretMode = 'new' | 'existing';

export interface NewSecretData {
  name: string;
  authMode: string;
  secretFields: Record<string, string>;
  configFields: Record<string, string>;
}

export interface ApiKeyConfiguration {
  mode: 'new' | 'existing';
  existingSecretId: string;
  newSecret: NewSecretData;
}
