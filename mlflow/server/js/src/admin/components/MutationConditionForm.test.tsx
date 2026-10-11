import { describe, it, expect, jest } from '@jest/globals';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';

import {
  MutationConditionForm,
  draftToStagedCondition,
  isMutationConditionDraftDirty,
  isMutationConditionDraftFillable,
  isPartialWildcardPattern,
  MUTATION_CONDITION_DRAFT_DEFAULT,
  type MutationConditionDraft,
} from './MutationConditionForm';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const draft = (overrides: Partial<MutationConditionDraft> = {}): MutationConditionDraft => ({
  ...MUTATION_CONDITION_DRAFT_DEFAULT,
  ...overrides,
});

describe('MutationConditionForm — draft translation', () => {
  it('refuses a draft with neither filter', () => {
    // The server rejects it too: an object with neither filter restricts nothing,
    // so it is not a state worth storing. Catching it here means the admin gets a
    // field-level reminder instead of a request error.
    expect(isMutationConditionDraftFillable(draft())).toBe(false);
  });

  it('accepts a draft with either filter alone', () => {
    expect(isMutationConditionDraftFillable(draft({ valueCondition: "tag_value != 'prod'" }))).toBe(true);
    expect(isMutationConditionDraftFillable(draft({ targetCondition: "tags.x = 'y'" }))).toBe(true);
  });

  it('treats whitespace as no filter at all', () => {
    // Otherwise a stray space submits, and the parser reports a syntax error for
    // what is really an empty field.
    expect(isMutationConditionDraftFillable(draft({ valueCondition: '   ' }))).toBe(false);
  });

  it('requires a parent once the scope is narrowed to one', () => {
    const scoped = draft({ scope: 'scoped', resourceType: 'run', targetCondition: "tags.x = 'y'" });
    expect(isMutationConditionDraftFillable(scoped)).toBe(false);
    expect(isMutationConditionDraftFillable({ ...scoped, scopePattern: '42' })).toBe(true);
  });

  it('sends an absent filter as null, never as an empty string', () => {
    // ``''`` would reach the condition parser and be reported as a filter syntax
    // error rather than as an absent filter.
    const request = draftToStagedCondition(draft({ targetCondition: "tags.x = 'y'" }));
    expect(request.valueCondition).toBeNull();
    expect(request.targetCondition).toBe("tags.x = 'y'");
  });

  it('puts the id on the axis the type actually narrows on', () => {
    // The draft carries one pattern; which of the server's two axes receives it is
    // derived from the type. Getting that wrong is a write the server rejects.
    const unscoped = draftToStagedCondition(draft({ resourceType: 'run', targetCondition: "tags.x = 'y'" }));
    expect(unscoped.resourcePattern).toBe('*');
    expect(unscoped.containerResourceType).toBe('workspace');
    expect(unscoped.containerResourcePattern).toBe('*');

    // A sub-resource narrows by its CONTAINER and stays wildcard itself.
    const scopedChild = draftToStagedCondition(
      draft({ resourceType: 'run', scope: 'scoped', scopePattern: '42', targetCondition: "tags.x = 'y'" }),
    );
    expect(scopedChild.resourcePattern).toBe('*');
    expect(scopedChild.containerResourceType).toBe('experiment');
    expect(scopedChild.containerResourcePattern).toBe('42');

    // A top-level type narrows by its OWN pattern and stays in the workspace.
    const scopedTop = draftToStagedCondition(
      draft({ resourceType: 'experiment', scope: 'scoped', scopePattern: '7', targetCondition: "tags.x = 'y'" }),
    );
    expect(scopedTop.resourcePattern).toBe('7');
    expect(scopedTop.containerResourceType).toBe('workspace');
    expect(scopedTop.containerResourcePattern).toBe('*');
  });

  it('derives the parent type from the resource type rather than trusting the draft', () => {
    // The draft never carries a parent type. If it did, it could disagree with the
    // resource type and persist a scope that can never match.
    const versionScoped = draftToStagedCondition(
      draft({
        resourceType: 'registered_model_version',
        scope: 'scoped',
        scopePattern: 'my-model',
        targetCondition: "tags.x = 'y'",
      }),
    );
    expect(versionScoped.containerResourceType).toBe('registered_model');
  });

  it('drops a parent scope the resource type cannot have', () => {
    // ``experiment`` is parentless. A scope left over from a previous selection
    // must not travel, or the server rejects the write for a field the admin can
    // no longer see.
    const request = draftToStagedCondition(
      draft({
        resourceType: 'experiment',
        scope: 'scoped',
        scopePattern: 'stale',
        targetCondition: "tags.x = 'y'",
      }),
    );
    expect(request.containerResourceType).toBe('workspace');
    expect(request.containerResourcePattern).toBe('*');
  });

  it('trims the filters it sends', () => {
    const request = draftToStagedCondition(draft({ targetCondition: "  tags.x = 'y'  " }));
    expect(request.targetCondition).toBe("tags.x = 'y'");
  });

  it('reports a default draft as clean and any edit as dirty', () => {
    expect(isMutationConditionDraftDirty(draft())).toBe(false);
    expect(isMutationConditionDraftDirty(draft({ targetCondition: 'x' }))).toBe(true);
    expect(isMutationConditionDraftDirty(draft({ resourceType: 'run' }))).toBe(true);
  });
});

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

describe('MutationConditionForm — the absence rule is stated on both fields', () => {
  // `tags.a = 'x'` is valid in BOTH fields and inverts on absence: it passes a request
  // that does not set `a`, and refuses a resource that does not have `a`. An admin who
  // copies a clause from one field to the other gets the opposite policy with no error,
  // so each field has to say which way it goes. These assertions are deliberately on the
  // DIRECTION words, not on the full sentences, so the wording can be improved without
  // the guarantee silently disappearing.
  const renderForm = () =>
    renderWithDesignSystem(
      <MutationConditionForm value={MUTATION_CONDITION_DRAFT_DEFAULT} onChange={jest.fn()} workspace="default" />,
    );

  it('says a value condition PASSES a request that omits the key', () => {
    renderForm();
    const hint = screen.getByTestId('admin.mutation_condition_form.value_condition_hint');
    expect(hint.textContent).toMatch(/does not set the key passes/i);
  });

  it('says a target condition REFUSES a resource that lacks the tag', () => {
    renderForm();
    const hint = screen.getByTestId('admin.mutation_condition_form.target_condition_hint');
    expect(hint.textContent).toMatch(/does not have the tag is refused/i);
  });

  it('warns that the same clause differs between the two fields', () => {
    renderForm();
    const hint = screen.getByTestId('admin.mutation_condition_form.target_condition_hint');
    expect(hint.textContent).toMatch(/opposite|different things/i);
  });

  // The rest of each field's explanation is hover help on the label, so that the two
  // absence rules above stay the shortest thing in the form and remain comparable at a
  // glance. These assertions open the tooltip the way a user does rather than reading the
  // prop, so a hint that renders but never reveals still fails.
  it('explains the keyed form on hover, since it is the only way to free other keys', async () => {
    renderForm();

    await userEvent.hover(screen.getByRole('img', { name: 'More information about value conditions' }));

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toContain("tags.a IN ('x','y')"));
  });

  it('explains on hover that a target condition never blocks a create', async () => {
    renderForm();

    await userEvent.hover(screen.getByRole('img', { name: 'More information about target conditions' }));

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toMatch(/never blocks a create/i));
  });

  it('reveals the hover help on keyboard focus, not only on hover', async () => {
    // The explanation is the only place the keyed form is documented in the UI, so it has
    // to be reachable without a pointer.
    renderForm();
    const icon = screen.getByRole('img', { name: 'More information about value conditions' });

    expect(icon).toHaveAttribute('tabindex', '0');
    fireEvent.focus(icon);

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toContain('tags.a IN'));
  });
});

describe('MutationConditionForm — a partial wildcard scope is refused', () => {
  // A scope pattern is matched EXACTLY server-side, so a glob governs nothing -- and a
  // condition that governs nothing does not withhold access, it silently fails to
  // restrict it. The server refuses it; this keeps the draft unfillable so the reason
  // appears next to the field instead of arriving as a rejected submit.

  it.each(['team/*', 'prefix*', '*suffix', 'a*b'])('treats %s as a partial wildcard', (glob) => {
    expect(isPartialWildcardPattern(glob)).toBe(true);
  });

  it.each(['*', 'team/alpha', '', '  '])('does not treat %s as one', (ok) => {
    expect(isPartialWildcardPattern(ok)).toBe(false);
  });

  it('refuses to fill a scoped draft whose pattern is a glob', () => {
    const base = draft({ valueCondition: "tag_key = 'env'", scope: 'scoped' as const });
    expect(isMutationConditionDraftFillable({ ...base, scopePattern: 'team/alpha' })).toBe(true);
    expect(isMutationConditionDraftFillable({ ...base, scopePattern: 'team/*' })).toBe(false);
  });

  it('still fills an all-scoped draft, which is what sends the bare wildcard', () => {
    // The guard must not catch the one wildcard that works: `All ...` sends `*`
    // without going through scopePattern at all.
    expect(isMutationConditionDraftFillable(draft({ valueCondition: "tag_key = 'env'", scope: 'all' as const }))).toBe(
      true,
    );
  });
});
