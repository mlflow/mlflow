import { describe, it, expect, jest } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DirectPermissionForm, DIRECT_PERMISSION_DEFAULT } from './DirectPermissionForm';

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

describe('DirectPermissionForm — permission picker filtering', () => {
  it('coerces permission to a gateway-grantable value when switching from experiment to gateway_secret', async () => {
    // ``USE`` is only meaningful on gateway resources; switching the other
    // direction (gateway → experiment with permission=USE) must coerce
    // the now-invalid USE down to READ. Cover that path too.
    const onChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionForm
        value={{ ...DIRECT_PERMISSION_DEFAULT, resourceType: 'gateway_secret', permission: 'USE' }}
        onChange={onChange}
      />,
    );
    const resourceTypeTrigger = document.getElementById('admin-direct-permission-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    await userEvent.click(await screen.findByRole('option', { name: 'Experiment' }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ resourceType: 'experiment', permission: 'READ' }));
  });

  it('does not offer gateway_model_definition (intentionally removed from DIRECT_GRANT_RESOURCE_TYPES)', async () => {
    renderWithDesignSystem(<DirectPermissionForm value={DIRECT_PERMISSION_DEFAULT} onChange={() => {}} />);
    const resourceTypeTrigger = document.getElementById('admin-direct-permission-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    expect(screen.queryByRole('option', { name: 'Gateway model definition' })).not.toBeInTheDocument();
  });

  it('offers prompt and scorer alongside the other direct-grant types', async () => {
    // Parity with the role picker minus workspace (which the backend
    // rejects on the per-user convenience API).
    renderWithDesignSystem(<DirectPermissionForm value={DIRECT_PERMISSION_DEFAULT} onChange={() => {}} />);
    const resourceTypeTrigger = document.getElementById('admin-direct-permission-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    expect(await screen.findByRole('option', { name: 'Prompt' })).toBeInTheDocument();
    expect(await screen.findByRole('option', { name: 'Scorer' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Workspace' })).not.toBeInTheDocument();
  });

  it('switches to All scope without writing the wildcard into resourceId', async () => {
    // The wildcard is derived from scope at staging time so a resource
    // literally named ``*`` can't masquerade as an all-of-type grant.
    const onChange = jest.fn();
    renderWithDesignSystem(<DirectPermissionForm value={DIRECT_PERMISSION_DEFAULT} onChange={onChange} />);
    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ scope: 'all', resourceId: '' }));
  });

  it('preserves the All scope when resource type changes', async () => {
    // The form is still submittable because ``isDirectPermissionSubmittable``
    // short-circuits on ``scope === 'all'`` regardless of ``resourceId``.
    const onChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionForm
        value={{ resourceType: 'experiment', scope: 'all', resourceId: '', permission: 'READ' }}
        onChange={onChange}
      />,
    );
    const resourceTypeTrigger = document.getElementById('admin-direct-permission-resource-type')!;
    await userEvent.click(resourceTypeTrigger);
    await userEvent.click(await screen.findByRole('option', { name: 'Registered model' }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ resourceType: 'registered_model', scope: 'all', resourceId: '' }),
    );
  });
});
