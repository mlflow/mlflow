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

export interface ModelConfiguration {
  id: string;
  provider: string;
  modelName: string;
  apiKey: ApiKeyConfiguration;
}

export interface ModelConfigurationErrors {
  provider?: string;
  modelName?: string;
  apiKey?: {
    mode?: string;
    existingSecretId?: string;
    newSecret?: {
      name?: string;
      secretFields?: Record<string, string>;
      configFields?: Record<string, string>;
    };
  };
}

export interface ModelConfigurationProps {
  value: ModelConfiguration;
  onChange: (value: ModelConfiguration) => void;
  errors?: ModelConfigurationErrors;
  disabled?: boolean;
  componentIdPrefix?: string;
}

export interface ModelConfigurationState {
  value: ModelConfiguration;
  isComplete: boolean;
  errors: ModelConfigurationErrors;
}

export interface ModelConfigurationActions {
  setValue: (value: ModelConfiguration) => void;
  setProvider: (provider: string) => void;
  setModelName: (modelName: string) => void;
  setApiKey: (apiKey: ApiKeyConfiguration) => void;
  reset: () => void;
}
