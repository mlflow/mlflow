import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { screen } from '@testing-library/react';
import { renderWithDesignSystem } from '../../../../../common/utils/TestUtils.react18';
import { GenAIApiKeyConfigurator } from './GenAIApiKeyConfigurator';
import { useSecretsQuery } from '../../../../../gateway/hooks/useSecretsQuery';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';

jest.mock('../../../../../gateway/hooks/useSecretsQuery');

const NON_ASCII_ERROR = /API key contains non-ASCII characters/;

const buildValue = (apiKey: string): ApiKeyConfiguration => ({
  mode: 'new',
  existingSecretId: '',
  newSecret: { name: 'my-key', authMode: 'api_key', secretFields: { api_key: apiKey }, configFields: {} },
});

const renderConfigurator = (value: ApiKeyConfiguration) =>
  renderWithDesignSystem(
    <GenAIApiKeyConfigurator
      value={value}
      onChange={jest.fn()}
      provider="openai"
      authModes={[{ mode: 'api_key', secret_fields: [{ name: 'api_key', required: true }], config_fields: [] } as any]}
      defaultAuthMode="api_key"
      isLoadingProviderConfig={false}
      hasExistingSecrets={false}
    />,
  );

describe('GenAIApiKeyConfigurator', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useSecretsQuery).mockReturnValue({ data: [] } as any);
  });

  test('explains why a single-field API key with non-ASCII characters is rejected', () => {
    renderConfigurator(buildValue('가나다'));
    expect(screen.getByText(NON_ASCII_ERROR)).toBeInTheDocument();
  });

  test('does not show the non-ASCII error for an ASCII API key', () => {
    renderConfigurator(buildValue('rkskek'));
    expect(screen.queryByText(NON_ASCII_ERROR)).not.toBeInTheDocument();
  });

  test('does not show the non-ASCII error while the input is searching saved key names', () => {
    jest
      .mocked(useSecretsQuery)
      .mockReturnValue({ data: [{ secret_id: 's-1', secret_name: '프로덕션-키', provider: 'openai' }] } as any);
    renderConfigurator(buildValue('프로덕'));
    expect(screen.queryByText(NON_ASCII_ERROR)).not.toBeInTheDocument();
  });
});
