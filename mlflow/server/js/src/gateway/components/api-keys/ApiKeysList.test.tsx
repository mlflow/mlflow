import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { ApiKeysList } from './ApiKeysList';
import { useSecretsQuery } from '../../hooks/useSecretsQuery';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { useBindingsQuery } from '../../hooks/useBindingsQuery';
import { useModelDefinitionsQuery } from '../../hooks/useModelDefinitionsQuery';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

jest.mock('../../hooks/useSecretsQuery');
jest.mock('../../hooks/useEndpointsQuery');
jest.mock('../../hooks/useBindingsQuery');
jest.mock('../../hooks/useModelDefinitionsQuery');

const mockSecrets = [
  {
    secret_id: 's-123',
    secret_name: 'openai-key',
    provider: 'openai',
    masked_values: { api_key: 'sk-****1234' },
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
  },
];

const mockEndpoints = [
  {
    endpoint_id: 'ep-123',
    name: 'test-endpoint',
    model_mappings: [
      {
        mapping_id: 'mm-1',
        endpoint_id: 'ep-123',
        model_definition_id: 'md-1',
        model_definition: {
          model_definition_id: 'md-1',
          name: 'test-model',
          secret_id: 's-123',
          secret_name: 'openai-key',
          provider: 'openai',
          model_name: 'gpt-4',
          created_at: Date.now(),
          last_updated_at: Date.now(),
        },
        weight: 1,
        created_at: Date.now(),
      },
    ],
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
  },
];

const mockModelDefinitions = [
  {
    model_definition_id: 'md-1',
    name: 'test-model',
    secret_id: 's-123',
    secret_name: 'openai-key',
    provider: 'openai',
    model_name: 'gpt-4',
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
  },
];

describe('ApiKeysList', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useBindingsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
  });

  test('renders loading state', () => {
    jest.mocked(useSecretsQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <ApiKeysList />
      </MemoryRouter>,
    );

    expect(screen.getByText('Loading API keys...')).toBeInTheDocument();
  });

  test('renders empty state when no secrets', () => {
    jest.mocked(useSecretsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <ApiKeysList />
      </MemoryRouter>,
    );

    expect(screen.getByText('No API keys created')).toBeInTheDocument();
  });

  test('renders secrets list', () => {
    jest.mocked(useSecretsQuery).mockReturnValue({
      data: mockSecrets,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <ApiKeysList />
      </MemoryRouter>,
    );

    expect(screen.getByText('openai-key')).toBeInTheDocument();
    expect(screen.getByText('OpenAI')).toBeInTheDocument();
  });

  test('filters secrets by search', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    jest.mocked(useSecretsQuery).mockReturnValue({
      data: [
        ...mockSecrets,
        {
          secret_id: 's-456',
          secret_name: 'anthropic-key',
          provider: 'anthropic',
          masked_values: { api_key: 'sk-****5678' },
          created_at: Date.now() / 1000,
          last_updated_at: Date.now() / 1000,
        },
      ],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <ApiKeysList />
      </MemoryRouter>,
    );

    const searchInput = screen.getByPlaceholderText('Search API Keys');
    await userEvent.type(searchInput, 'anthropic');

    expect(screen.getByText('anthropic-key')).toBeInTheDocument();
    expect(screen.queryByText('openai-key')).not.toBeInTheDocument();
  });

  test('shows model count for secrets', () => {
    jest.mocked(useSecretsQuery).mockReturnValue({
      data: mockSecrets,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: mockModelDefinitions,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <ApiKeysList />
      </MemoryRouter>,
    );

    expect(screen.getByText('1')).toBeInTheDocument();
  });

  test('calls onKeyClick when secret name is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onKeyClick = jest.fn();

    jest.mocked(useSecretsQuery).mockReturnValue({
      data: mockSecrets,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <ApiKeysList onKeyClick={onKeyClick} />
      </MemoryRouter>,
    );

    await userEvent.click(screen.getByText('openai-key'));

    expect(onKeyClick).toHaveBeenCalledWith(mockSecrets[0]);
  });
});
