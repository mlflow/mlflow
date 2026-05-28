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

describe('DirectPermissionsSection — unsaved-invalid-draft signal', () => {
  it('reports unsaved-invalid=true when the user picks scope=specific with no resource', async () => {
    // The exact silent-drop case: admin bumps the permission level to MANAGE
    // (changes from the default READ) but leaves scope on the default
    // ``specific`` and never picks a specific resource. Section must
    // surface this state so the parent modal can block submit.
    mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
    const onUnsavedInvalidDraftChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionsSection
        value={[]}
        onChange={jest.fn()}
        onUnsavedInvalidDraftChange={onUnsavedInvalidDraftChange}
      />,
    );
    // ``waitFor`` because the reporting effect fires after the first
    // commit, not synchronously inside ``render`` — otherwise a flaky
    // schedule could let the assertion pass vacuously before the mock is
    // ever called.
    await waitFor(() => expect(onUnsavedInvalidDraftChange).toHaveBeenLastCalledWith(false));

    await changeSimpleSelect('admin.direct_permission.permission_level', 'MANAGE');

    await waitFor(() => expect(onUnsavedInvalidDraftChange).toHaveBeenLastCalledWith(true));

    // Surfaces the inline error so the admin knows why the modal is locked.
    expect(screen.getByTestId('admin.direct_permission.resource_required_error')).toBeInTheDocument();
  });

  it('flips back to unsaved-invalid=false when the user clears the draft', async () => {
    // ``Clear`` is the escape hatch: if the admin decides not to add a
    // direct grant after all, resetting the draft to default unlocks the
    // parent modal's submit.
    mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
    const onUnsavedInvalidDraftChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionsSection
        value={[]}
        onChange={jest.fn()}
        onUnsavedInvalidDraftChange={onUnsavedInvalidDraftChange}
      />,
    );
    await changeSimpleSelect('admin.direct_permission.permission_level', 'MANAGE');
    await waitFor(() => expect(onUnsavedInvalidDraftChange).toHaveBeenLastCalledWith(true));

    await userEvent.click(screen.getByRole('button', { name: /^Clear$/ }));
    await waitFor(() => expect(onUnsavedInvalidDraftChange).toHaveBeenLastCalledWith(false));
    // Inline error disappears once the draft is clean.
    expect(screen.queryByTestId('admin.direct_permission.resource_required_error')).not.toBeInTheDocument();
  });

  it('does not flag scope=all as unsaved-invalid (a wildcard grant is submittable on its own)', async () => {
    // The "specific resource missing" warning is exclusive to scope=specific
    // — once the user flips to ``All experiments``, the draft is fully
    // submittable, so the section reports false and the inline error stays
    // hidden even though the draft is dirty.
    mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
    const onUnsavedInvalidDraftChange = jest.fn();
    renderWithDesignSystem(
      <DirectPermissionsSection
        value={[]}
        onChange={jest.fn()}
        onUnsavedInvalidDraftChange={onUnsavedInvalidDraftChange}
      />,
    );
    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    await waitFor(() => expect(onUnsavedInvalidDraftChange).toHaveBeenLastCalledWith(false));
    expect(screen.queryByTestId('admin.direct_permission.resource_required_error')).not.toBeInTheDocument();
  });
});
