import { describe, it, expect, jest } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RolePermissionForm, ROLE_PERMISSION_DRAFT_DEFAULT } from './RolePermissionForm';

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

describe('RolePermissionForm — permission picker filtering', () => {
  it('coerces permission to a workspace-grantable value when resource type switches to workspace', async () => {
    // Workspace scope only accepts USE/MANAGE — READ would 400 on submit.
    const onChange = jest.fn();
    renderWithDesignSystem(
      <RolePermissionForm value={{ ...ROLE_PERMISSION_DRAFT_DEFAULT, permission: 'READ' }} onChange={onChange} />,
    );
    const resourceTypeTrigger = document.getElementById('admin-role-permission-form-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    await userEvent.click(await screen.findByRole('option', { name: 'workspace' }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ resourceType: 'workspace', permission: 'USE' }));
  });
});
