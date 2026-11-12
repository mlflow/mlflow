import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { SecretFormFields } from './SecretFormFields';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';

jest.mock('../../hooks/useProviderConfigQuery');

const NON_ASCII_ERROR = /API key contains non-ASCII characters/;

describe('SecretFormFields', () => {
  beforeEach(() => {
    jest.mocked(useProviderConfigQuery).mockReturnValue({
      data: {
        default_mode: 'default',
        auth_modes: [
          {
            mode: 'default',
            display_name: 'Default',
            secret_fields: [
              { name: 'api_key', required: true },
              { name: 'vertex_credentials', required: false },
            ],
            config_fields: [],
          },
        ],
      },
      isLoading: false,
    } as any);
  });

  test('shows the non-ASCII error for API key fields', () => {
    renderWithDesignSystem(
      <SecretFormFields
        provider="vertex_ai"
        value={{ name: 'key', authMode: 'default', secretFields: { api_key: '가나다' }, configFields: {} }}
        onChange={jest.fn()}
      />,
    );
    expect(screen.getByText(NON_ASCII_ERROR)).toBeInTheDocument();
  });

  test('does not show the non-ASCII error for JSON credential fields', () => {
    renderWithDesignSystem(
      <SecretFormFields
        provider="vertex_ai"
        value={{
          name: 'key',
          authMode: 'default',
          secretFields: { api_key: 'sk-abc123', vertex_credentials: '{"client_email": "서비스@example.com"}' },
          configFields: {},
        }}
        onChange={jest.fn()}
      />,
    );
    expect(screen.queryByText(NON_ASCII_ERROR)).not.toBeInTheDocument();
  });
});
