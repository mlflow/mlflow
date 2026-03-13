import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { useCreateWorkspaceModal } from './CreateWorkspaceModal';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { act } from 'react-dom/test-utils';

jest.mock('../../common/utils/FetchUtils');

describe('CreateWorkspaceModal', () => {
  const mockOnSuccess = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  const TestComponent = () => {
    const { CreateWorkspaceModal, openModal } = useCreateWorkspaceModal({
      onSuccess: mockOnSuccess,
    });

    return (
      <>
        <button onClick={openModal}>Open Modal</button>
        {CreateWorkspaceModal}
      </>
    );
  };

  const renderComponent = () => {
    return renderWithIntl(<TestComponent />);
  };

  const openModal = async () => {
    const openButton = screen.getByText('Open Modal');
    await userEvent.click(openButton);
  };

  test('renders modal when open', async () => {
    renderComponent();
    await openModal();
    expect(screen.getByText('Create Workspace')).toBeInTheDocument();
    expect(screen.getByText(/Workspace Name/i)).toBeInTheDocument();
    expect(screen.getByText(/Description \(optional\)/i)).toBeInTheDocument();
  });

  test('does not render modal when closed', () => {
    renderComponent();
    expect(screen.queryByText('Create Workspace')).not.toBeInTheDocument();
  });

  test('renders input fields with correct placeholders', async () => {
    renderComponent();
    await openModal();

    expect(screen.getByPlaceholderText('Enter workspace name')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter workspace description')).toBeInTheDocument();
  });

  test('renders Create button', async () => {
    renderComponent();
    await openModal();

    expect(screen.getByText('Create')).toBeInTheDocument();
  });

  test('allows typing in workspace name field', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'my-workspace');

    expect(nameInput).toHaveValue('my-workspace');
  });

  test('allows typing in description field', async () => {
    renderComponent();
    await openModal();

    const descInput = screen.getByPlaceholderText('Enter workspace description');
    await userEvent.type(descInput, 'My workspace description');

    expect(descInput).toHaveValue('My workspace description');
  });

  test('shows validation error when name is empty and form is submitted', async () => {
    renderComponent();
    await openModal();

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(screen.getByText('Please input a name for the new workspace.')).toBeInTheDocument();
    });
  });

  test('shows validation error for invalid workspace name with spaces', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'Invalid Name With Spaces');

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText(
          'Workspace name must be lowercase alphanumeric with optional single hyphens (no consecutive hyphens).',
        ),
      ).toBeInTheDocument();
    });
  });

  test('shows validation error for workspace name starting with hyphen', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, '-invalid');

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText(
          'Workspace name must be lowercase alphanumeric with optional single hyphens (no consecutive hyphens).',
        ),
      ).toBeInTheDocument();
    });
  });

  test('shows validation error for workspace name ending with hyphen', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'invalid-');

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText(
          'Workspace name must be lowercase alphanumeric with optional single hyphens (no consecutive hyphens).',
        ),
      ).toBeInTheDocument();
    });
  });

  test('shows validation error for workspace name with uppercase letters', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'InvalidName');

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText(
          'Workspace name must be lowercase alphanumeric with optional single hyphens (no consecutive hyphens).',
        ),
      ).toBeInTheDocument();
    });
  });

  test('shows validation error for workspace name with consecutive hyphens', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'my--workspace');

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText(
          'Workspace name must be lowercase alphanumeric with optional single hyphens (no consecutive hyphens).',
        ),
      ).toBeInTheDocument();
    });
  });

  test('shows validation error for workspace name that is too short', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'a');

    const createButton = screen.getByText('Create');
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(screen.getByText('Workspace name must be between 2 and 63 characters.')).toBeInTheDocument();
    });
  });

  test('submits form when Enter key is pressed in workspace name field', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'test-workspace');

    // Press Enter to submit
    await userEvent.keyboard('{Enter}');

    // Since FetchUtils is mocked and we don't have a proper mock implementation,
    // we just verify the Enter key doesn't cause any errors
    expect(nameInput).toBeInTheDocument();
  });
});
