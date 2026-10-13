import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import { waitFor } from '@testing-library/react';
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

// ``SimpleSelect`` renders as a radix combobox button (no native ``<select>``);
// open it, then click the desired option. Same opens-listbox-and-clicks
// pattern works for all of the form's SimpleSelects (resource type,
// permission level).
const changeSimpleSelect = async (componentId: string, optionLabel: string) => {
  const trigger = document.querySelector<HTMLElement>(`[data-component-id="${componentId}"]`);
  if (!trigger) throw new Error(`SimpleSelect "${componentId}" not found`);
  await userEvent.click(trigger);
  await userEvent.click(await screen.findByRole('option', { name: optionLabel }));
};

describe('DirectPermissionsSection — unsaved-draft signal', () => {
  it('reports unsaved-draft=true for the dirty + scope=specific + no-resource case, and renders the inline error', async () => {
    // Admin bumps the permission level to MANAGE (changes from the default
    // READ) but leaves scope on the default ``specific`` and never picks a
    // specific resource. ``onUnsavedDraftChange`` must fire ``true`` so the
    // parent modal can pop the discard-confirm dialog on submit, and the
    // inline "Select a specific X" error must render to tell the admin
    // what's missing.
    mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
    const onUnsavedDraftChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionsSection value={[]} onChange={jest.fn()} onUnsavedDraftChange={onUnsavedDraftChange} />,
    );
    // ``waitFor`` because the reporting effect fires after the first
    // commit, not synchronously inside ``render`` — otherwise a flaky
    // schedule could let the assertion pass vacuously before the mock is
    // ever called.
    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenLastCalledWith(false));

    await changeSimpleSelect('admin.direct_permission.permission_level', 'MANAGE');

    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenLastCalledWith(true));
    expect(screen.getByTestId('admin.direct_permission.resource_required_error')).toBeInTheDocument();
  });

  it('reports unsaved-draft=true for scope=all (dirty + submittable, parent still wants the discard prompt)', async () => {
    // Picking "All experiments" is a complete, stage-able draft — but if
    // the admin then clicks submit without ``Add``, the all-grant is
    // silently dropped. ``onUnsavedDraftChange`` must fire ``true`` so the
    // parent modal can pop the discard-confirm dialog. The inline error
    // stays hidden because "Select a specific X" doesn't apply when the
    // user has already chosen the wildcard scope.
    mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
    const onUnsavedDraftChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionsSection value={[]} onChange={jest.fn()} onUnsavedDraftChange={onUnsavedDraftChange} />,
    );
    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenLastCalledWith(true));
    expect(screen.queryByTestId('admin.direct_permission.resource_required_error')).not.toBeInTheDocument();
  });

  it('flips back to unsaved-draft=false when the user clears the draft', async () => {
    // ``Clear`` is the escape hatch: if the admin decides not to add a
    // direct grant after all, resetting the draft to default silences the
    // discard prompt without forcing a click-through.
    mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
    const onUnsavedDraftChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionsSection value={[]} onChange={jest.fn()} onUnsavedDraftChange={onUnsavedDraftChange} />,
    );
    await changeSimpleSelect('admin.direct_permission.permission_level', 'MANAGE');
    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenLastCalledWith(true));

    await userEvent.click(screen.getByRole('button', { name: /^Clear$/ }));
    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenLastCalledWith(false));
    expect(screen.queryByTestId('admin.direct_permission.resource_required_error')).not.toBeInTheDocument();
  });
});
