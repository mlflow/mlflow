import { describe, expect, jest, test } from '@jest/globals';
import { FormProvider, useForm } from 'react-hook-form';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { EndpointFormRenderer, type EndpointFormData } from './EndpointFormRenderer';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import { useApiKeyConfiguration } from '../model-configuration/hooks/useApiKeyConfiguration';
import { UsageTrackingConfigurator } from '../edit-endpoint/UsageTrackingConfigurator';

jest.mock('../../hooks/useModelsQuery');
jest.mock('../../hooks/useProviderConfigQuery');
jest.mock('../model-configuration/hooks/useApiKeyConfiguration');
jest.mock('../create-endpoint', () => ({ ProviderSelect: () => null }));
jest.mock('../create-endpoint/ModelSelect', () => ({ ModelSelect: () => null }));
jest.mock('../edit-endpoint/UsageTrackingConfigurator', () => ({ UsageTrackingConfigurator: jest.fn(() => null) }));

const authModes = [{ mode: 'api_key', secret_fields: [{ name: 'api_key', required: true }], config_fields: [] }];

describe('EndpointFormRenderer subscriptions', () => {
  test('keeps unrelated sections stable while updating credentials, summary and submission', async () => {
    jest.mocked(useModelsQuery).mockReturnValue({ data: [] } as any);
    jest.mocked(useProviderConfigQuery).mockReturnValue({ data: { auth_modes: authModes } } as any);
    jest.mocked(useApiKeyConfiguration).mockReturnValue({
      existingSecrets: [],
      isLoadingSecrets: false,
      authModes,
      defaultAuthMode: 'api_key',
      isLoadingProviderConfig: false,
    } as any);
    const onSubmit = jest.fn(async (_values: EndpointFormData) => {});
    const resetErrors = jest.fn();
    function Harness() {
      const form = useForm<EndpointFormData>({
        defaultValues: {
          name: 'test-endpoint',
          provider: 'openai',
          modelName: 'model-1',
          secretMode: 'new',
          existingSecretId: '',
          newSecret: { name: '', authMode: 'api_key', secretFields: {}, configFields: {} },
          usageTracking: false,
        },
      });
      return (
        <MemoryRouter>
          <FormProvider {...form}>
            <EndpointFormRenderer
              mode="create"
              isSubmitting={false}
              error={null}
              errorMessage={null}
              resetErrors={resetErrors}
              onSubmit={onSubmit}
              onCancel={jest.fn()}
              onNameBlur={jest.fn()}
            />
          </FormProvider>
        </MemoryRouter>
      );
    }
    renderWithDesignSystem(<Harness />);
    jest.mocked(UsageTrackingConfigurator).mockClear();
    const create = screen.getByRole('button', { name: 'Create' });
    expect(create).toBeDisabled();
    await userEvent.type(screen.getByRole('textbox', { name: /API key name/ }), 'my-key');
    await userEvent.type(screen.getByLabelText(/^API Key/), 'synthetic-key');
    await waitFor(() => expect(create).toBeEnabled());
    expect(screen.getByText('my-key')).toBeInTheDocument();
    await userEvent.type(screen.getByPlaceholderText('my-endpoint'), '-updated');
    expect(resetErrors).not.toHaveBeenCalled();
    expect(UsageTrackingConfigurator).not.toHaveBeenCalled();
    await userEvent.click(create);
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      name: 'test-endpoint-updated',
      newSecret: { name: 'my-key', secretFields: { api_key: 'synthetic-key' } },
    });
  });
});
