import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { EndpointsList } from './EndpointsList';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { useDeleteEndpointMutation } from '../../hooks/useDeleteEndpointMutation';
import { useBindingsQuery } from '../../hooks/useBindingsQuery';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

jest.mock('../../hooks/useEndpointsQuery');
jest.mock('../../hooks/useDeleteEndpointMutation');
jest.mock('../../hooks/useBindingsQuery');

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
          name: 'test-model-def',
          secret_id: 's-1',
          secret_name: 'test-secret',
          provider: 'openai',
          model_name: 'gpt-4',
          created_at: Date.now(),
          last_updated_at: Date.now(),
        },
        weight: 1,
        created_at: Date.now(),
      },
    ],
    created_at: Date.now(),
    last_updated_at: Date.now(),
  },
];

describe('EndpointsList', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useDeleteEndpointMutation).mockReturnValue({
      mutate: jest.fn(),
      isLoading: false,
    } as any);
    jest.mocked(useBindingsQuery).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);
  });

  test('renders loading state', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <EndpointsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('Loading endpoints...')).toBeInTheDocument();
  });

  test('renders empty state when no endpoints', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <EndpointsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('No endpoints created')).toBeInTheDocument();
  });

  test('renders endpoints list', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <EndpointsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('test-endpoint')).toBeInTheDocument();
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
  });

  test('filters endpoints by search', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const { waitFor } = await import('@testing-library/react');

    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [
        ...mockEndpoints,
        {
          endpoint_id: 'ep-456',
          name: 'another-endpoint',
          model_mappings: [
            {
              mapping_id: 'mm-2',
              endpoint_id: 'ep-456',
              model_definition_id: 'md-2',
              model_definition: {
                model_definition_id: 'md-2',
                name: 'claude-model-def',
                secret_id: 's-2',
                secret_name: 'anthropic-secret',
                provider: 'anthropic',
                model_name: 'claude-3',
                created_at: Date.now(),
                last_updated_at: Date.now(),
              },
              weight: 1,
              created_at: Date.now(),
            },
          ],
          created_at: Date.now(),
          last_updated_at: Date.now(),
        },
      ],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <EndpointsList />
      </MemoryRouter>,
    );

    const searchInput = screen.getByPlaceholderText('Search Endpoints');
    await userEvent.type(searchInput, 'another');

    // Wait for debounce (250ms) to complete
    await waitFor(() => {
      expect(screen.queryByText('test-endpoint')).not.toBeInTheDocument();
    });
    expect(screen.getByText('another-endpoint')).toBeInTheDocument();
  });
});
