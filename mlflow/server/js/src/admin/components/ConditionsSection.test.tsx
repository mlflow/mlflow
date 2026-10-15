import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { conditionKey, formatStagedCondition, ConditionsSection, type StagedCondition } from './ConditionsSection';
import { CONDITION_TYPE_VALUE, CONDITION_TYPE_TARGET } from '../types';

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

const onChange = jest.fn();

/**
 * Pick the condition type. The form carries one filter field whose vocabulary, hint and
 * placeholder all follow this, so a test about target-side behaviour has to select it
 * rather than relying on a second field existing.
 */
const selectTargetType = async () => {
  await userEvent.click(screen.getByRole('combobox', { name: 'Condition type' }));
  await userEvent.click(screen.getByRole('option', { name: 'Resources that may be mutated' }));
};

describe('ConditionsSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('stages the filter as typed, with Add gating a blank one', async () => {
    // "" reaches the condition parser and comes back as a filter syntax error, so the
    // absent filter has to be null by the time it leaves the form.
    renderWithDesignSystem(<ConditionsSection value={[]} onChange={onChange} />);
    await userEvent.type(screen.getByPlaceholderText("tag_value != 'prod'"), "tags.x = 'y'");
    await userEvent.click(screen.getByRole('button', { name: 'Add condition' }));

    await waitFor(() => expect(onChange).toHaveBeenCalledTimes(1));
    expect(onChange.mock.calls[0][0]).toEqual([
      expect.objectContaining({
        resourceType: 'experiment',
        type: CONDITION_TYPE_VALUE,
        filter: "tags.x = 'y'",
        resourcePattern: '*',
        containerResourceType: 'workspace',
        containerResourcePattern: '*',
      }),
    ]);
  });

  it('keeps Add disabled until a filter is entered', async () => {
    renderWithDesignSystem(<ConditionsSection value={[]} onChange={onChange} />);
    const add = screen.getByRole('button', { name: 'Add condition' });
    expect(add).toBeDisabled();
    await userEvent.type(screen.getByPlaceholderText("tag_value != 'prod'"), "tags.x = 'y'");
    await waitFor(() => expect(add).toBeEnabled());
  });

  it('lists staged rows and removes one by index', async () => {
    const value: StagedCondition[] = [
      {
        id: 5,
        resourceType: 'run',
        resourcePattern: '*',
        containerResourceType: 'workspace',
        containerResourcePattern: '*',
        type: CONDITION_TYPE_TARGET,
        filter: "tags.lifecycle != 'prod'",
      },
    ];
    renderWithDesignSystem(<ConditionsSection value={value} onChange={onChange} />);
    expect(screen.getByText("tags.lifecycle != 'prod'")).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Remove Run condition/ }));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it('collapses an exact duplicate instead of staging it twice', async () => {
    const value: StagedCondition[] = [
      {
        resourceType: 'experiment',
        resourcePattern: '*',
        containerResourceType: 'workspace',
        containerResourcePattern: '*',
        type: CONDITION_TYPE_VALUE,
        filter: "tags.x = 'y'",
      },
    ];
    renderWithDesignSystem(<ConditionsSection value={value} onChange={onChange} />);
    await userEvent.type(screen.getByPlaceholderText("tag_value != 'prod'"), "tags.x = 'y'");
    await userEvent.click(screen.getByRole('button', { name: 'Add condition' }));
    // Draft is cleared, but nothing is appended.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Add condition' })).toBeDisabled());
    expect(onChange).not.toHaveBeenCalled();
  });

  it('reports a dirty draft so the parent can gate its discard dialog', async () => {
    const onUnsavedDraftChange = jest.fn();
    renderWithDesignSystem(
      <ConditionsSection value={[]} onChange={onChange} onUnsavedDraftChange={onUnsavedDraftChange} />,
    );
    await userEvent.type(screen.getByPlaceholderText("tag_value != 'prod'"), 'x');
    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenCalledWith(true));
  });

  it('describes a staged condition for the review step', () => {
    expect(
      formatStagedCondition({
        resourceType: 'trace',
        resourcePattern: '*',
        containerResourceType: 'experiment',
        containerResourcePattern: '42',
        type: CONDITION_TYPE_VALUE,
        filter: "tag_value != 'prod'",
      }),
    ).toBe("trace [Experiment 42] value: tag_value != 'prod'");
  });

  it('offers a scope to a top-level type, which had none before', () => {
    // The gap this model closes: an experiment has no container, so under the old
    // parent-only scope its conditions could only ever cover the whole workspace.
    renderWithDesignSystem(<ConditionsSection value={[]} onChange={onChange} />);
    expect(screen.getByText('Only one experiment')).toBeInTheDocument();
  });
});

describe('conditionKey', () => {
  // Shared by both edit modals, which each used to keep their own copy. Both copies had
  // dropped `resourcePattern`, so two conditions on the same type differing only in scope
  // produced the same key: the diff could not tell them apart and a removal of either was
  // silently dropped, leaving the admin unable to lift a restriction.

  const base: StagedCondition = {
    resourceType: 'experiment',
    resourcePattern: '*',
    containerResourceType: 'workspace',
    containerResourcePattern: '*',
    type: CONDITION_TYPE_TARGET,
    filter: "tags.lifecycle != 'prod'",
  };

  it('distinguishes two conditions differing only in resource scope', () => {
    expect(conditionKey({ ...base, resourcePattern: '7' })).not.toBe(conditionKey(base));
  });

  it('distinguishes two conditions differing only in container scope', () => {
    const inWorkspace: StagedCondition = { ...base, resourceType: 'run' };
    const inOneExperiment: StagedCondition = {
      ...inWorkspace,
      containerResourceType: 'experiment',
      containerResourcePattern: '42',
    };
    expect(conditionKey(inOneExperiment)).not.toBe(conditionKey(inWorkspace));
  });

  it('is stable for the same condition and ignores the row id', () => {
    // The id is server-assigned and absent on a staged row, so it must not take part --
    // otherwise a prefilled row never matches the staged row it came from.
    expect(conditionKey({ ...base, id: 9 })).toBe(conditionKey(base));
  });

  it('distinguishes the two filter kinds', () => {
    const asValue: StagedCondition = { ...base, type: CONDITION_TYPE_VALUE, filter: "tag_key != 'pii'" };
    const asTarget: StagedCondition = { ...base, type: CONDITION_TYPE_TARGET, filter: "tag_key != 'pii'" };
    expect(conditionKey(asValue)).not.toBe(conditionKey(asTarget));
  });
});

describe("ConditionsSection — the type's vocabulary gates staging", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('refuses to stage an alias clause on a type that owns no aliases', async () => {
    // The default type is Experiment, which owns no aliases, so the backend would refuse
    // this condition. Blocking it here is the difference between a field-level message and
    // a staged row that reads as configured policy until submit fails.
    renderWithDesignSystem(<ConditionsSection value={[]} onChange={onChange} />);
    await userEvent.type(screen.getByPlaceholderText("tag_value != 'prod'"), "alias != 'champion'");

    const add = screen.getByRole('button', { name: 'Add condition' });
    await waitFor(() => expect(add).toBeDisabled());
    expect(screen.getByTestId('admin.condition_form.filter_vocabulary_error')).toHaveTextContent(
      "'alias' is not available on Experiment",
    );

    await userEvent.click(add);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('refuses a flat request identifier in a target condition', async () => {
    // The two types accept overlapping syntax, so a clause carried from one to the other
    // looks right; `tag_key` has no resource-side meaning.
    renderWithDesignSystem(<ConditionsSection value={[]} onChange={onChange} />);
    await selectTargetType();
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tag_key != 'pii'");

    await waitFor(() => expect(screen.getByRole('button', { name: 'Add condition' })).toBeDisabled());
    expect(screen.getByTestId('admin.condition_form.filter_vocabulary_error')).toHaveTextContent(
      "'tag_key' is not available on Experiment",
    );
  });

  it("stages the condition once the clause uses the type's vocabulary", async () => {
    renderWithDesignSystem(<ConditionsSection value={[]} onChange={onChange} />);
    const field = screen.getByPlaceholderText("tag_value != 'prod'");
    await userEvent.type(field, "alias != 'champion'");
    await waitFor(() => expect(screen.getByRole('button', { name: 'Add condition' })).toBeDisabled());

    await userEvent.clear(field);
    await userEvent.type(field, "tag_value != 'prod'");
    const add = screen.getByRole('button', { name: 'Add condition' });
    await waitFor(() => expect(add).toBeEnabled());
    expect(screen.queryByTestId('admin.condition_form.filter_vocabulary_error')).not.toBeInTheDocument();

    await userEvent.click(add);
    await waitFor(() => expect(onChange).toHaveBeenCalledTimes(1));
    expect(onChange.mock.calls[0][0]).toEqual([
      expect.objectContaining({
        resourceType: 'experiment',
        type: CONDITION_TYPE_VALUE,
        filter: "tag_value != 'prod'",
      }),
    ]);
  });
});
