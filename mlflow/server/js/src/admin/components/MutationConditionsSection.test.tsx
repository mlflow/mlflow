import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import {
  conditionKey,
  formatStagedCondition,
  MutationConditionsSection,
  type StagedMutationCondition,
} from './MutationConditionsSection';

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

const onChange = jest.fn();

describe('MutationConditionsSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('stages an empty filter as null rather than an empty string', async () => {
    // "" reaches the condition parser and comes back as a filter syntax error, so the
    // absent filter has to be null by the time it leaves the form.
    renderWithDesignSystem(<MutationConditionsSection value={[]} onChange={onChange} />);
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.x = 'y'");
    await userEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));

    await waitFor(() => expect(onChange).toHaveBeenCalledTimes(1));
    expect(onChange.mock.calls[0][0]).toEqual([
      expect.objectContaining({
        resourceType: 'experiment',
        valueCondition: null,
        targetCondition: "tags.x = 'y'",
        resourcePattern: '*',
        containerResourceType: 'workspace',
        containerResourcePattern: '*',
      }),
    ]);
  });

  it('keeps Add disabled until a filter is entered', async () => {
    renderWithDesignSystem(<MutationConditionsSection value={[]} onChange={onChange} />);
    const add = screen.getByRole('button', { name: 'Add mutation condition' });
    expect(add).toBeDisabled();
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.x = 'y'");
    await waitFor(() => expect(add).toBeEnabled());
  });

  it('lists staged rows and removes one by index', async () => {
    const value: StagedMutationCondition[] = [
      {
        id: 5,
        resourceType: 'run',
        resourcePattern: '*',
        containerResourceType: 'workspace',
        containerResourcePattern: '*',
        valueCondition: null,
        targetCondition: "tags.lifecycle != 'prod'",
      },
    ];
    renderWithDesignSystem(<MutationConditionsSection value={value} onChange={onChange} />);
    expect(screen.getByText("tags.lifecycle != 'prod'")).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Remove Run mutation condition/ }));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it('collapses an exact duplicate instead of staging it twice', async () => {
    const value: StagedMutationCondition[] = [
      {
        resourceType: 'experiment',
        resourcePattern: '*',
        containerResourceType: 'workspace',
        containerResourcePattern: '*',
        valueCondition: null,
        targetCondition: "tags.x = 'y'",
      },
    ];
    renderWithDesignSystem(<MutationConditionsSection value={value} onChange={onChange} />);
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.x = 'y'");
    await userEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));
    // Draft is cleared, but nothing is appended.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Add mutation condition' })).toBeDisabled());
    expect(onChange).not.toHaveBeenCalled();
  });

  it('reports a dirty draft so the parent can gate its discard dialog', async () => {
    const onUnsavedDraftChange = jest.fn();
    renderWithDesignSystem(
      <MutationConditionsSection value={[]} onChange={onChange} onUnsavedDraftChange={onUnsavedDraftChange} />,
    );
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), 'x');
    await waitFor(() => expect(onUnsavedDraftChange).toHaveBeenCalledWith(true));
  });

  it('describes a staged condition for the review step', () => {
    expect(
      formatStagedCondition({
        resourceType: 'trace',
        resourcePattern: '*',
        containerResourceType: 'experiment',
        containerResourcePattern: '42',
        valueCondition: "tag_value != 'prod'",
        targetCondition: "tags.reviewed = 'yes'",
      }),
    ).toBe("trace [Experiment 42] value: tag_value != 'prod', target: tags.reviewed = 'yes'");
  });

  it('offers a scope to a top-level type, which had none before', () => {
    // The gap this model closes: an experiment has no container, so under the old
    // parent-only scope its conditions could only ever cover the whole workspace.
    renderWithDesignSystem(<MutationConditionsSection value={[]} onChange={onChange} />);
    expect(screen.getByText('Only one experiment')).toBeInTheDocument();
  });
});

describe('conditionKey', () => {
  // Shared by both edit modals, which each used to keep their own copy. Both copies had
  // dropped `resourcePattern`, so two conditions on the same type differing only in scope
  // produced the same key: the diff could not tell them apart and a removal of either was
  // silently dropped, leaving the admin unable to lift a restriction.

  const base: StagedMutationCondition = {
    resourceType: 'experiment',
    resourcePattern: '*',
    containerResourceType: 'workspace',
    containerResourcePattern: '*',
    valueCondition: null,
    targetCondition: "tags.lifecycle != 'prod'",
  };

  it('distinguishes two conditions differing only in resource scope', () => {
    expect(conditionKey({ ...base, resourcePattern: '7' })).not.toBe(conditionKey(base));
  });

  it('distinguishes two conditions differing only in container scope', () => {
    const inWorkspace: StagedMutationCondition = { ...base, resourceType: 'run' };
    const inOneExperiment: StagedMutationCondition = {
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
    const asValue: StagedMutationCondition = { ...base, valueCondition: "tag_key != 'pii'", targetCondition: null };
    const asTarget: StagedMutationCondition = { ...base, valueCondition: null, targetCondition: "tag_key != 'pii'" };
    expect(conditionKey(asValue)).not.toBe(conditionKey(asTarget));
  });
});
