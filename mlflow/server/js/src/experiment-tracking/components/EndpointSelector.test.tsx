import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen, within } from '../../common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { EndpointSelector } from './EndpointSelector';
import { useEndpointsQuery } from '../../gateway/hooks/useEndpointsQuery';
import type { Endpoint } from '../../gateway/types';

jest.mock('../../gateway/hooks/useEndpointsQuery');
jest.mock('../../gateway/components/endpoint-form', () => ({
  CreateEndpointModal: ({ open, onClose }: { open: boolean; onClose: () => void }) =>
    open ? <div data-testid="create-endpoint-modal">Create Endpoint Modal</div> : null,
}));

const mockEndpoints: Endpoint[] = [
  {
    endpoint_id: 'ep-1',
    name: 'openai-endpoint',
    created_at: 1700000000000,
    last_updated_at: 1700000000000,
    model_mappings: [
      {
        mapping_id: 'mm-1',
        endpoint_id: 'ep-1',
        model_definition_id: 'md-1',
        weight: 1,
        created_at: 1700000000000,
        model_definition: {
          model_definition_id: 'md-1',
          name: 'model-1',
          provider: 'openai',
          model_name: 'gpt-4',
          secret_id: 'secret-1',
          secret_name: 'secret-1',
          created_at: 1700000000000,
          last_updated_at: 1700000000000,
          endpoint_count: 1,
        },
      },
    ],
  },
  {
    endpoint_id: 'ep-2',
    name: 'anthropic-endpoint',
    created_at: 1700000000000,
    last_updated_at: 1700000000000,
    model_mappings: [
      {
        mapping_id: 'mm-2',
        endpoint_id: 'ep-2',
        model_definition_id: 'md-2',
        weight: 1,
        created_at: 1700000000000,
        model_definition: {
          model_definition_id: 'md-2',
          name: 'model-2',
          provider: 'anthropic',
          model_name: 'claude-3',
          secret_id: 'secret-2',
          secret_name: 'secret-2',
          created_at: 1700000000000,
          last_updated_at: 1700000000000,
          endpoint_count: 1,
        },
      },
    ],
  },
];

describe('EndpointSelector', () => {
  const mockOnEndpointSelect = jest.fn();
  const mockRefetch = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: true,
      error: undefined,
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(<EndpointSelector onEndpointSelect={mockOnEndpointSelect} />);

    expect(screen.getByText('Loading endpoints...')).toBeInTheDocument();
  });

  test('renders error state', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: new Error('Failed to fetch'),
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(<EndpointSelector onEndpointSelect={mockOnEndpointSelect} />);

    expect(screen.getByText('Failed to fetch')).toBeInTheDocument();
  });

  test('renders endpoints in dropdown and calls onEndpointSelect when selected', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(<EndpointSelector onEndpointSelect={mockOnEndpointSelect} />);

    // Open dropdown
    await userEvent.click(screen.getByRole('combobox'));

    // Verify endpoints are shown
    const listbox = screen.getByRole('listbox');
    expect(within(listbox).getByText('openai-endpoint')).toBeInTheDocument();
    expect(within(listbox).getByText('anthropic-endpoint')).toBeInTheDocument();

    // Verify provider/model hints are shown
    expect(within(listbox).getByText('openai / gpt-4')).toBeInTheDocument();
    expect(within(listbox).getByText('anthropic / claude-3')).toBeInTheDocument();

    // Select an endpoint
    await userEvent.click(within(listbox).getByText('anthropic-endpoint'));

    expect(mockOnEndpointSelect).toHaveBeenCalledWith('anthropic-endpoint');
  });

  test('opens create endpoint modal when clicking create button', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(<EndpointSelector onEndpointSelect={mockOnEndpointSelect} />);

    // Open dropdown
    await userEvent.click(screen.getByRole('combobox'));

    // Click create button
    await userEvent.click(screen.getByText('Create new endpoint'));

    // Modal should be visible
    expect(screen.getByTestId('create-endpoint-modal')).toBeInTheDocument();
  });

  test('displays current endpoint name when provided', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(
      <EndpointSelector currentEndpointName="openai-endpoint" onEndpointSelect={mockOnEndpointSelect} />,
    );

    expect(screen.getByText('openai-endpoint')).toBeInTheDocument();
    expect(screen.getByText('(openai / gpt-4)')).toBeInTheDocument();
  });

  test('shows warning for deleted endpoint', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(
      <EndpointSelector currentEndpointName="deleted-endpoint" onEndpointSelect={mockOnEndpointSelect} />,
    );

    // Should show the endpoint name even though it's not in the list
    expect(screen.getByText('deleted-endpoint')).toBeInTheDocument();
  });

  test('dropdown does not open when disabled', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: mockRefetch,
    } as any);

    renderWithDesignSystem(<EndpointSelector onEndpointSelect={mockOnEndpointSelect} disabled />);

    // Combobox should be rendered but listbox should not be present
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });
});
