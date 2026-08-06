import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RolePermissionForm, ROLE_PERMISSION_DRAFT_DEFAULT } from './RolePermissionForm';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: jest.fn(),
}));

const mockUseWorkspacesEnabled = jest.mocked(useWorkspacesEnabled);
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

beforeEach(() => {
  mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: true, loading: false });
});

describe('RolePermissionForm — permission picker filtering', () => {
  it('coerces permission to a workspace-grantable value when resource type switches to workspace', async () => {
    // Workspace scope only accepts USE/MANAGE — READ would 400 on submit.
    const onChange = jest.fn();
    renderWithDesignSystem(
      <RolePermissionForm value={{ ...ROLE_PERMISSION_DRAFT_DEFAULT, permission: 'READ' }} onChange={onChange} />,
    );
    const resourceTypeTrigger = document.getElementById('admin-role-permission-form-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    await userEvent.click(await screen.findByRole('option', { name: 'Workspace' }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ resourceType: 'workspace', permission: 'USE' }));
  });

  it('does not offer gateway_model_definition (intentionally removed from RESOURCE_TYPES)', async () => {
    renderWithDesignSystem(<RolePermissionForm value={ROLE_PERMISSION_DRAFT_DEFAULT} onChange={() => {}} />);
    const resourceTypeTrigger = document.getElementById('admin-role-permission-form-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    expect(screen.queryByRole('option', { name: 'gateway_model_definition' })).not.toBeInTheDocument();
  });

  it('hides workspace from the dropdown in single-tenant mode', async () => {
    mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: false, loading: false });
    renderWithDesignSystem(<RolePermissionForm value={ROLE_PERMISSION_DRAFT_DEFAULT} onChange={() => {}} />);
    const resourceTypeTrigger = document.getElementById('admin-role-permission-form-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    expect(screen.queryByRole('option', { name: 'Workspace' })).not.toBeInTheDocument();
    expect(await screen.findByRole('option', { name: 'Experiment' })).toBeInTheDocument();
  });

  it('keeps workspace visible while the server-info query is loading', async () => {
    // Same no-flicker default used by other admin views: defer hiding
    // until we're sure the server is single-tenant.
    mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: false, loading: true });
    renderWithDesignSystem(<RolePermissionForm value={ROLE_PERMISSION_DRAFT_DEFAULT} onChange={() => {}} />);
    const resourceTypeTrigger = document.getElementById('admin-role-permission-form-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    expect(await screen.findByRole('option', { name: 'Workspace' })).toBeInTheDocument();
  });

  it('renders the prompt picker with the same Specific/All shape as other types', async () => {
    renderWithDesignSystem(
      <RolePermissionForm
        value={{ ...ROLE_PERMISSION_DRAFT_DEFAULT, resourceType: 'prompt', scope: 'specific' }}
        onChange={() => {}}
      />,
    );
    // Specific scope renders a DialogCombobox-backed picker (combobox role),
    // not a freetext ``<input>``. Asserting the combobox is present is the
    // strong claim — absence of freetext is implied.
    expect(screen.getByRole('combobox', { name: /Prompt, no option selected/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /^All prompts$/ })).toBeInTheDocument();
  });
});
