import type { ReactNode } from 'react';
import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';

import { AssistantSettingsForm } from './AssistantSettingsForm';
import * as AssistantService from '../AssistantService';
import type { AssistantConfig } from '../types';

let mockIsLocalServer = true;
const mockRefetchConfig = jest.fn();
let mockConfig: AssistantConfig | null = null;

jest.mock('../AssistantContext', () => ({
  useAssistant: () => ({ isLocalServer: mockIsLocalServer }),
}));

jest.mock('../hooks/useAssistantConfigQuery', () => ({
  useAssistantConfigQuery: () => ({
    config: mockConfig,
    isLoading: false,
    refetch: mockRefetchConfig,
  }),
}));

jest.mock('../AssistantService', () => ({
  __esModule: true,
  updateConfig: jest.fn(() => Promise.resolve({})),
  installSkills: jest.fn(() => Promise.resolve({})),
}));

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  Link: ({ children }: { children: ReactNode }) => <a href="#">{children}</a>,
}));

const mockUpdateConfig = jest.mocked(AssistantService.updateConfig);
const mockInstallSkills = jest.mocked(AssistantService.installSkills);

const renderForm = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <AssistantSettingsForm experimentId="123" provider="claude_code" onBack={jest.fn()} onComplete={jest.fn()} />
    </DesignSystemProvider>,
  );

describe('AssistantSettingsForm', () => {
  beforeEach(() => {
    mockUpdateConfig.mockClear();
    mockInstallSkills.mockClear();
    mockRefetchConfig.mockClear();
    mockIsLocalServer = true;
    mockConfig = {
      providers: {
        claude_code: {
          model: 'default',
          selected: true,
          permissions: { allow_edit_files: true, allow_read_docs: true, full_access: true },
        },
      },
      projects: { '123': { type: 'local', location: '/srv/project' } },
    };
  });

  test('a remote client sees host-only fields disabled and the local paths replaced with a note', () => {
    mockIsLocalServer = false;
    renderForm();

    // Provider permissions the remote caller CAN set are still editable.
    expect(screen.getByRole('checkbox', { name: /Read MLflow doc/ })).not.toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /Edit project code/ })).not.toBeDisabled();

    // Full access cannot be granted from a remote client.
    expect(screen.getByRole('checkbox', { name: /Full access/ })).toBeDisabled();

    // Project path and skills point at the host filesystem, so the inputs are replaced by a note.
    expect(screen.queryByPlaceholderText('/Users/me/projects/my-llm-project')).not.toBeInTheDocument();
    expect(screen.getByText(/Project paths point at the MLflow server host/)).toBeInTheDocument();
    expect(screen.queryByText('Custom location')).not.toBeInTheDocument();
    expect(screen.getByText(/Skills are installed on the MLflow server host/)).toBeInTheDocument();
  });

  test('a remote save writes only provider settings, forces full access off, and skips skills install', async () => {
    const user = userEvent.setup();
    mockIsLocalServer = false;
    renderForm();

    await user.click(screen.getByRole('button', { name: 'Finish' }));

    await waitFor(() => expect(mockUpdateConfig).toHaveBeenCalledTimes(1));
    const payload = mockUpdateConfig.mock.calls[0][0];
    expect(payload.providers?.['claude_code'].permissions?.full_access).toBe(false);
    // Project mappings are host-only, so a remote save must not touch them.
    expect(payload.projects).toBeUndefined();
    // Skills install targets the host filesystem and must not run for a remote caller.
    expect(mockInstallSkills).not.toHaveBeenCalled();
  });

  test('a local save configures the project mapping and installs skills', async () => {
    const user = userEvent.setup();
    renderForm();

    // The local host can grant full access.
    expect(screen.getByRole('checkbox', { name: /Full access/ })).not.toBeDisabled();
    expect(screen.getByPlaceholderText('/Users/me/projects/my-llm-project')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Finish' }));

    await waitFor(() => expect(mockUpdateConfig).toHaveBeenCalledTimes(1));
    const payload = mockUpdateConfig.mock.calls[0][0];
    expect(payload.providers?.['claude_code'].permissions?.full_access).toBe(true);
    expect(payload.projects?.['123']).toEqual({ type: 'local', location: '/srv/project' });
    expect(mockInstallSkills).toHaveBeenCalledTimes(1);
  });

  test('re-enables Finish after a save error once a permission is changed', async () => {
    const user = userEvent.setup();
    mockUpdateConfig.mockRejectedValueOnce(new Error('Full access can only be enabled from the MLflow server host.'));
    renderForm();

    await user.click(screen.getByRole('button', { name: 'Finish' }));

    // The save failed: the error is shown and Finish is disabled.
    await waitFor(() => expect(screen.getByText(/Full access can only be enabled/)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Finish' })).toBeDisabled();

    // Changing a permission checkbox (not only a path input) clears the error and re-enables
    // Finish, so the user is not stuck after a 403.
    await user.click(screen.getByRole('checkbox', { name: /Read MLflow doc/ }));
    expect(screen.queryByText(/Full access can only be enabled/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Finish' })).not.toBeDisabled();
  });
});
