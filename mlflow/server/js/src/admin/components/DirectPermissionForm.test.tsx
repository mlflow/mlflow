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
});
