import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { BulkDeleteApiKeyModal } from './BulkDeleteApiKeyModal';
import { useDeleteSecret } from '../../hooks/useDeleteSecret';
import type { Endpoint } from '../../types';

jest.mock('../../hooks/useDeleteSecret');

const mockSecrets = [
  {
    secret_id: 's-1',
    secret_name: 'openai-key',
    provider: 'openai',
    masked_values: { api_key: 'sk-****1234' },
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
  },
  {
    secret_id: 's-2',
    secret_name: 'anthropic-key',
    provider: 'anthropic',
    masked_values: { api_key: 'sk-****5678' },
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
  },
];

describe('BulkDeleteApiKeyModal', () => {
  const mockDeleteSecret = jest.fn<(secretId: string) => Promise<void>>();
  const mockOnClose = jest.fn();
  const mockOnSuccess = jest.fn();
  const mockGetEndpointsForSecret = jest.fn<(secretId: string) => Endpoint[]>().mockReturnValue([]);

  beforeEach(() => {
    jest.clearAllMocks();
    mockDeleteSecret.mockResolvedValue(undefined);
    mockGetEndpointsForSecret.mockReturnValue([]);
    jest.mocked(useDeleteSecret).mockReturnValue({
      mutateAsync: mockDeleteSecret,
    } as any);
  });

  test('renders list of secrets to delete', () => {
    renderWithDesignSystem(
      <BulkDeleteApiKeyModal
        open
        secrets={mockSecrets}
        getEndpointsForSecret={mockGetEndpointsForSecret}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    expect(screen.getByText('openai-key')).toBeInTheDocument();
    expect(screen.getByText('anthropic-key')).toBeInTheDocument();
  });

  test('shows warning when secrets are used by endpoints', () => {
    mockGetEndpointsForSecret.mockImplementation((secretId: string) =>
      secretId === 's-1' ? ([{ endpoint_id: 'ep-1', name: 'test-endpoint' }] as Endpoint[]) : [],
    );

    renderWithDesignSystem(
      <BulkDeleteApiKeyModal
        open
        secrets={mockSecrets}
        getEndpointsForSecret={mockGetEndpointsForSecret}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    expect(screen.getByText(/1 API key is currently in use by endpoints/)).toBeInTheDocument();
  });

  test('calls deleteSecret for each secret on confirm', async () => {
    renderWithDesignSystem(
      <BulkDeleteApiKeyModal
        open
        secrets={mockSecrets}
        getEndpointsForSecret={mockGetEndpointsForSecret}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(mockDeleteSecret).toHaveBeenCalledTimes(2);
      expect(mockDeleteSecret).toHaveBeenCalledWith('s-1');
      expect(mockDeleteSecret).toHaveBeenCalledWith('s-2');
      expect(mockOnSuccess).toHaveBeenCalled();
    });
  });

  test('shows error when deletion fails', async () => {
    mockDeleteSecret.mockRejectedValue(new Error('Network error'));

    renderWithDesignSystem(
      <BulkDeleteApiKeyModal
        open
        secrets={mockSecrets}
        getEndpointsForSecret={mockGetEndpointsForSecret}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(screen.getByText('Failed to delete some API keys. Please try again.')).toBeInTheDocument();
    });
    // onSuccess is still called so the list refreshes (some keys may have been deleted)
    expect(mockOnSuccess).toHaveBeenCalled();
    // Modal stays open to show the error
    expect(mockOnClose).not.toHaveBeenCalled();
  });

  test('calls onClose when cancel is clicked', async () => {
    renderWithDesignSystem(
      <BulkDeleteApiKeyModal
        open
        secrets={mockSecrets}
        getEndpointsForSecret={mockGetEndpointsForSecret}
        onClose={mockOnClose}
        onSuccess={mockOnSuccess}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(mockOnClose).toHaveBeenCalled();
  });
});
