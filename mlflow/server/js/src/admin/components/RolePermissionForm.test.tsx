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

describe('RolePermissionForm — wildcard-only resource types', () => {
  // The backend TYPE grain map gives a sub-resource tier PatternKind.WILDCARD
  // alone, so a grant can never name one row. The picker must not offer a scope
  // the backend would reject.
  it.each([
    ['run', 'Run'],
    ['trace', 'Trace'],
    ['assessment', 'Assessment'],
    ['logged_model', 'Logged model'],
    ['review_queue', 'Review queue'],
    ['registered_model_version', 'Model version'],
    ['prompt_version', 'Prompt version'],
    ['scorer_version', 'Scorer version'],
    ['mcp_server_version', 'MCP server version'],
  ])('disables the specific-resource scope for %s', (resourceType, label) => {
    renderWithDesignSystem(
      <RolePermissionForm
        value={{ ...ROLE_PERMISSION_DRAFT_DEFAULT, resourceType, scope: 'all' }}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole('radio', { name: new RegExp(`^Specific ${label.toLowerCase()}$`) })).toBeDisabled();
    expect(screen.getByRole('radio', { name: new RegExp(`^All ${label.toLowerCase()}s$`) })).toBeEnabled();
  });

  it('leaves the specific scope selectable for an id-capable type', () => {
    renderWithDesignSystem(
      <RolePermissionForm
        value={{ ...ROLE_PERMISSION_DRAFT_DEFAULT, resourceType: 'experiment', scope: 'all' }}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole('radio', { name: /^Specific experiment$/ })).toBeEnabled();
  });

  it('offers every wildcard-only tier in the type dropdown', async () => {
    renderWithDesignSystem(<RolePermissionForm value={ROLE_PERMISSION_DRAFT_DEFAULT} onChange={() => {}} />);
    await userEvent.click(document.getElementById('admin-role-permission-form-resource-type')!);
    for (const label of ['Run', 'Trace', 'Assessment', 'Logged model', 'Review queue']) {
      expect(await screen.findByRole('option', { name: label })).toBeInTheDocument();
    }
  });

  it('offers DENY last in the permission dropdown', async () => {
    renderWithDesignSystem(<RolePermissionForm value={ROLE_PERMISSION_DRAFT_DEFAULT} onChange={() => {}} />);
    await userEvent.click(document.getElementById('admin-role-permission-form-level')!);
    const options = await screen.findAllByRole('option');
    expect(options.map((o) => o.textContent)).toEqual(['READ', 'EDIT', 'MANAGE', 'DENY']);
  });
});
