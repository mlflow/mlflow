import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { ModelDefinitionsList } from './ModelDefinitionsList';
import { useModelDefinitionsQuery } from '../../hooks/useModelDefinitionsQuery';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

jest.mock('../../hooks/useModelDefinitionsQuery');
jest.mock('../../hooks/useEndpointsQuery');

const mockModelDefinitions = [
  {
    model_definition_id: 'md-123',
    name: 'gpt-4-model',
    secret_id: 's-1',
    secret_name: 'openai-key',
    provider: 'openai',
    model_name: 'gpt-4',
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
    endpoint_count: 1,
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
        model_definition_id: 'md-123',
        weight: 1,
        created_at: Date.now(),
      },
    ],
    created_at: Date.now(),
    last_updated_at: Date.now(),
  },
];

describe('ModelDefinitionsList', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state', () => {
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
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

    renderWithDesignSystem(
      <MemoryRouter>
        <ModelDefinitionsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('Loading models...')).toBeInTheDocument();
  });

  test('renders empty state when no model definitions', () => {
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
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
        <ModelDefinitionsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('No models created yet')).toBeInTheDocument();
  });

  test('renders model definitions list', () => {
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: mockModelDefinitions,
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
        <ModelDefinitionsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('gpt-4-model')).toBeInTheDocument();
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
    expect(screen.getByText('openai-key')).toBeInTheDocument();
  });

  test('shows endpoint count for model definitions', () => {
    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: mockModelDefinitions,
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
        <ModelDefinitionsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('1')).toBeInTheDocument();
  });

  test('calls onModelDefinitionClick when model name is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onModelDefinitionClick = jest.fn();

    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: mockModelDefinitions,
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
        <ModelDefinitionsList onModelDefinitionClick={onModelDefinitionClick} />
      </MemoryRouter>,
    );

    await userEvent.click(screen.getByText('gpt-4-model'));

    expect(onModelDefinitionClick).toHaveBeenCalledWith(mockModelDefinitions[0]);
  });

  test('filters model definitions by search', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    jest.mocked(useModelDefinitionsQuery).mockReturnValue({
      data: [
        ...mockModelDefinitions,
        {
          model_definition_id: 'md-456',
          name: 'claude-model',
          secret_id: 's-2',
          secret_name: 'anthropic-key',
          provider: 'anthropic',
          model_name: 'claude-3-opus',
          created_at: Date.now() / 1000,
          last_updated_at: Date.now() / 1000,
          endpoint_count: 0,
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
        <ModelDefinitionsList />
      </MemoryRouter>,
    );

    const searchInput = screen.getByPlaceholderText('Search models');
    await userEvent.type(searchInput, 'claude');

    // Wait for debounce (250ms) to take effect
    await waitFor(() => {
      expect(screen.getByText('claude-model')).toBeInTheDocument();
      expect(screen.queryByText('gpt-4-model')).not.toBeInTheDocument();
    });
  });
});
