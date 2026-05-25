import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DirectPermissionsSection } from './DirectPermissionsSection';

const mockUseResourceOptionsQuery = jest.fn();
jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => mockUseResourceOptionsQuery(),
}));

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

beforeEach(() => {
  mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
});

describe('DirectPermissionsSection — staging layer', () => {
  it('translates scope=all into resourceId=* on Add (not at the draft layer)', async () => {
    // ``resourceId`` is the API-bound value; the staged grant must carry
    // ``'*'`` even though the draft never wrote it. Keeps a resource
    // literally named ``*`` from shadowing the wildcard at the form layer.
    const onChange = jest.fn();
    renderWithDesignSystem(<DirectPermissionsSection value={[]} onChange={onChange} />);
    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    expect(onChange).toHaveBeenCalledWith([expect.objectContaining({ resourceType: 'experiment', resourceId: '*' })]);
  });

  it('round-trips a specific picked id without rewriting it to the wildcard', async () => {
    // Complements the ``All`` test above: when scope stays ``specific`` and
    // the user picks an option, the staged grant carries that exact id —
    // proving the wildcard translation only fires on ``scope === 'all'``
    // and trusting the picker-layer ``*``-filter (see ``hooks.test.tsx``).
    mockUseResourceOptionsQuery.mockReturnValue({
      options: [
        { id: 'fraud', name: 'fraud' },
        { id: 'churn', name: 'churn' },
      ],
      isLoading: false,
      error: null,
    });
    const onChange = jest.fn();
    renderWithDesignSystem(<DirectPermissionsSection value={[]} onChange={onChange} />);
    // Multiple ``combobox`` roles exist in the form (resource type / id /
    // permission selects); aria-label disambiguates this one.
    await userEvent.click(screen.getByRole('combobox', { name: /Experiment, no option selected/ }));
    await userEvent.click(await screen.findByRole('option', { name: 'fraud' }));
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    expect(onChange).toHaveBeenCalledWith([
      expect.objectContaining({ resourceType: 'experiment', resourceId: 'fraud' }),
    ]);
  });
});
