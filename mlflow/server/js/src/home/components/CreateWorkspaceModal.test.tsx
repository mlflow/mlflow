import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { useCreateWorkspaceModal } from './CreateWorkspaceModal';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { fetchAPI } from '../../common/utils/FetchUtils';

jest.mock('../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(() => Promise.resolve({})),
  getAjaxUrl: jest.fn((url: string) => url),
  HTTPMethods: { GET: 'GET', POST: 'POST', PATCH: 'PATCH', DELETE: 'DELETE' },
}));

describe('CreateWorkspaceModal', () => {
  const mockOnSuccess = jest.fn();
  const mockFetchAPI = jest.mocked(fetchAPI);

  beforeEach(() => {
    jest.clearAllMocks();
    mockFetchAPI.mockResolvedValue({} as any);
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

  const openTraceArchivalSection = async () => {
    await userEvent.click(screen.getByRole('button', { name: 'Trace archival settings' }));
  };

  test('renders modal when open', async () => {
    renderComponent();
    await openModal();
    await openTraceArchivalSection();
    expect(screen.getByText('Create Workspace')).toBeInTheDocument();
    expect(screen.getByText(/Workspace Name/i)).toBeInTheDocument();
    expect(screen.getByText(/^Description$/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        'Optional. Override where archived trace payloads are stored for this workspace. Leave blank to use the server default.',
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Optional. Override how long traces stay in the tracking store before archival. Leave blank to use the server default.',
      ),
    ).toBeInTheDocument();
  });

  test('does not render modal when closed', () => {
    renderComponent();
    expect(screen.queryByText('Create Workspace')).not.toBeInTheDocument();
  });

  test('renders input fields with correct placeholders', async () => {
    renderComponent();
    await openModal();
    await openTraceArchivalSection();

    expect(screen.getByPlaceholderText('Enter workspace name')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter workspace description')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter default artifact root URI')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter trace archival location URI')).toBeInTheDocument();
    expect(screen.getByLabelText('Trace Archival Retention')).toBeInTheDocument();
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

  test('shows validation error for invalid trace archival retention', async () => {
    renderComponent();
    await openModal();
    await openTraceArchivalSection();

    await userEvent.type(screen.getByPlaceholderText('Enter workspace name'), 'team-a');
    await userEvent.type(screen.getByLabelText('Trace Archival Retention'), '30days');

    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(
        screen.getByText(
          "Trace archival retention must use the format <int><unit>, where unit is one of 'm', 'h', or 'd'.",
        ),
      ).toBeInTheDocument();
    });
    expect(mockFetchAPI).not.toHaveBeenCalled();
  });

  test('submits trace archival config when provided', async () => {
    renderComponent();
    await openModal();
    await openTraceArchivalSection();

    await userEvent.type(screen.getByPlaceholderText('Enter workspace name'), 'team-a');
    await userEvent.type(screen.getByPlaceholderText('Enter workspace description'), 'Team A workspace');
    await userEvent.type(screen.getByPlaceholderText('Enter default artifact root URI'), 's3://artifacts/team-a');
    await userEvent.type(screen.getByPlaceholderText('Enter trace archival location URI'), 's3://archive/team-a');
    await userEvent.type(screen.getByLabelText('Trace Archival Retention'), '30');

    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(mockFetchAPI).toHaveBeenCalledTimes(1);
    });

    const [url, options] = mockFetchAPI.mock.calls[0] as [string, { body: string; headers?: Record<string, string> }];
    expect(url).toBe('ajax-api/3.0/mlflow/workspaces');
    expect(JSON.parse(options.body)).toEqual({
      name: 'team-a',
      description: 'Team A workspace',
      default_artifact_root: 's3://artifacts/team-a',
      trace_archival_config: {
        location: 's3://archive/team-a',
        retention: '30d',
      },
    });
    expect(options.headers).toEqual({ 'X-MLFLOW-WORKSPACE': '' });
  });

  test('submits form when Enter key is pressed in workspace name field', async () => {
    renderComponent();
    await openModal();

    const nameInput = screen.getByPlaceholderText('Enter workspace name');
    await userEvent.type(nameInput, 'test-workspace');

    await userEvent.keyboard('{Enter}');

    expect(nameInput).toBeInTheDocument();
  });
});
