import { describe, test, jest, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import { DeleteConfirmationModal } from './DeleteConfirmationModal';

describe('DeleteConfirmationModal', () => {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    onConfirm: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    title: 'Delete Item',
    itemName: 'test-item',
    itemType: 'item',
    componentIdPrefix: 'test.delete-modal',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders modal with correct content', () => {
    renderWithIntl(<DeleteConfirmationModal {...defaultProps} />);

    expect(screen.getByText('Delete Item')).toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to delete/)).toBeInTheDocument();
    expect(screen.getByText('test-item')).toBeInTheDocument();
  });

  test('calls onConfirm and onClose when delete button clicked', async () => {
    const user = userEvent.setup();
    renderWithIntl(<DeleteConfirmationModal {...defaultProps} />);

    await user.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(defaultProps.onConfirm).toHaveBeenCalled();
    });
  });

  test('calls onClose when cancel button clicked', async () => {
    const user = userEvent.setup();
    renderWithIntl(<DeleteConfirmationModal {...defaultProps} />);

    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  test('shows warning message when provided', () => {
    renderWithIntl(<DeleteConfirmationModal {...defaultProps} warningMessage="This action cannot be undone" />);

    expect(screen.getByText('This action cannot be undone')).toBeInTheDocument();
  });

  test('requires confirmation input when requireConfirmation is true', async () => {
    const user = userEvent.setup();
    renderWithIntl(<DeleteConfirmationModal {...defaultProps} requireConfirmation />);

    const deleteButton = screen.getByRole('button', { name: 'Delete' });
    expect(deleteButton).toBeDisabled();

    const input = screen.getByPlaceholderText('test-item');
    await user.type(input, 'test-item');

    expect(deleteButton).not.toBeDisabled();
  });

  test('shows error message when onConfirm fails', async () => {
    const user = userEvent.setup();
    const onConfirmError = jest.fn<() => Promise<void>>().mockRejectedValue(new Error('Delete failed'));

    renderWithIntl(<DeleteConfirmationModal {...defaultProps} onConfirm={onConfirmError} />);

    await user.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(screen.getByText(/Failed to delete item/)).toBeInTheDocument();
    });
  });

  test('does not render when open is false', () => {
    renderWithIntl(<DeleteConfirmationModal {...defaultProps} open={false} />);

    expect(screen.queryByText('Delete Item')).not.toBeInTheDocument();
  });
});
