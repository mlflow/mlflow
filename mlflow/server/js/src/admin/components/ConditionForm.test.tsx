import { describe, it, expect, jest } from '@jest/globals';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';

import {
  ConditionForm,
  draftToStagedCondition,
  isConditionDraftDirty,
  isConditionDraftFillable,
  isPartialWildcardPattern,
  CONDITION_DRAFT_DEFAULT,
  type ConditionDraft,
} from './ConditionForm';
import { CONDITION_TYPE_VALUE, CONDITION_TYPE_TARGET } from '../types';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const draft = (overrides: Partial<ConditionDraft> = {}): ConditionDraft => ({
  ...CONDITION_DRAFT_DEFAULT,
  ...overrides,
});

describe('ConditionForm — draft translation', () => {
  it('refuses a draft with neither filter', () => {
    // The server rejects it too: an object with neither filter restricts nothing,
    // so it is not a state worth storing. Catching it here means the admin gets a
    // field-level reminder instead of a request error.
    expect(isConditionDraftFillable(draft())).toBe(false);
  });

  it('accepts a draft with either filter alone', () => {
    expect(isConditionDraftFillable(draft({ type: CONDITION_TYPE_VALUE, filter: "tag_value != 'prod'" }))).toBe(true);
    expect(isConditionDraftFillable(draft({ type: CONDITION_TYPE_TARGET, filter: "tags.x = 'y'" }))).toBe(true);
  });

  it('treats whitespace as no filter at all', () => {
    // Otherwise a stray space submits, and the parser reports a syntax error for
    // what is really an empty field.
    expect(isConditionDraftFillable(draft({ type: CONDITION_TYPE_VALUE, filter: '   ' }))).toBe(false);
  });

  it('requires a parent once the scope is narrowed to one', () => {
    const scoped = draft({ scope: 'scoped', resourceType: 'run', type: CONDITION_TYPE_TARGET, filter: "tags.x = 'y'" });
    expect(isConditionDraftFillable(scoped)).toBe(false);
    expect(isConditionDraftFillable({ ...scoped, scopePattern: '42' })).toBe(true);
  });

  it('sends the chosen type alongside its filter', () => {
    // There is no absent-filter state to normalise any more: one condition carries one
    // type and one filter, and a blank filter never reaches here because
    // isConditionDraftFillable gates the Add button on it.
    const request = draftToStagedCondition(draft({ type: CONDITION_TYPE_TARGET, filter: "tags.x = 'y'" }));
    expect(request.type).toBe(CONDITION_TYPE_TARGET);
    expect(request.filter).toBe("tags.x = 'y'");
  });

  it('puts the id on the axis the type actually narrows on', () => {
    // The draft carries one pattern; which of the server's two axes receives it is
    // derived from the type. Getting that wrong is a write the server rejects.
    const unscoped = draftToStagedCondition(
      draft({ resourceType: 'run', type: CONDITION_TYPE_TARGET, filter: "tags.x = 'y'" }),
    );
    expect(unscoped.resourcePattern).toBe('*');
    expect(unscoped.containerResourceType).toBe('workspace');
    expect(unscoped.containerResourcePattern).toBe('*');

    // A sub-resource narrows by its CONTAINER and stays wildcard itself.
    const scopedChild = draftToStagedCondition(
      draft({
        resourceType: 'run',
        scope: 'scoped',
        scopePattern: '42',
        type: CONDITION_TYPE_TARGET,
        filter: "tags.x = 'y'",
      }),
    );
    expect(scopedChild.resourcePattern).toBe('*');
    expect(scopedChild.containerResourceType).toBe('experiment');
    expect(scopedChild.containerResourcePattern).toBe('42');

    // A top-level type narrows by its OWN pattern and stays in the workspace.
    const scopedTop = draftToStagedCondition(
      draft({
        resourceType: 'experiment',
        scope: 'scoped',
        scopePattern: '7',
        type: CONDITION_TYPE_TARGET,
        filter: "tags.x = 'y'",
      }),
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
        type: CONDITION_TYPE_TARGET,
        filter: "tags.x = 'y'",
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
        type: CONDITION_TYPE_TARGET,
        filter: "tags.x = 'y'",
      }),
    );
    expect(request.containerResourceType).toBe('workspace');
    expect(request.containerResourcePattern).toBe('*');
  });

  it('trims the filter it sends', () => {
    const request = draftToStagedCondition(draft({ type: CONDITION_TYPE_TARGET, filter: "  tags.x = 'y'  " }));
    expect(request.filter).toBe("tags.x = 'y'");
  });

  it('reports a default draft as clean and any edit as dirty', () => {
    expect(isConditionDraftDirty(draft())).toBe(false);
    expect(isConditionDraftDirty(draft({ type: CONDITION_TYPE_TARGET, filter: 'x' }))).toBe(true);
    expect(isConditionDraftDirty(draft({ resourceType: 'run' }))).toBe(true);
  });
});

jest.mock('../hooks', () => ({
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
}));

describe('ConditionForm — the absence rule is stated for the selected type', () => {
  // `tags.a = 'x'` is valid under BOTH types and inverts on absence: it passes a request
  // that does not set `a`, and refuses a resource that does not have `a`. An admin who
  // carries a clause from one type to the other gets the opposite policy with no error,
  // so the field has to say which way the selected type goes. These assertions are
  // deliberately on the DIRECTION words, not the full sentences, so the wording can be
  // improved without the guarantee silently disappearing.
  //
  // There is one field now rather than two, so the rule is asserted per selected type
  // instead of per field, and the comparison between them has to be stated in the help
  // rather than shown by two fields sitting side by side.
  const renderForm = (type: ConditionDraft['type'] = CONDITION_TYPE_VALUE) =>
    renderWithDesignSystem(
      <ConditionForm value={{ ...CONDITION_DRAFT_DEFAULT, type }} onChange={jest.fn()} workspace="default" />,
    );

  it('says a value condition PASSES a request that omits the key', () => {
    renderForm(CONDITION_TYPE_VALUE);
    const hint = screen.getByTestId('admin.condition_form.filter_hint');
    expect(hint.textContent).toMatch(/does not set the key passes/i);
  });

  it('says a target condition REFUSES a resource that lacks the tag', () => {
    renderForm(CONDITION_TYPE_TARGET);
    const hint = screen.getByTestId('admin.condition_form.filter_hint');
    expect(hint.textContent).toMatch(/does not have the tag is refused/i);
  });

  it('warns that the same clause differs between the two types', () => {
    renderForm(CONDITION_TYPE_TARGET);
    const hint = screen.getByTestId('admin.condition_form.filter_hint');
    expect(hint.textContent).toMatch(/opposite|different things/i);
  });

  // The rest of the explanation is hover help on the labels, so the absence rule above
  // stays the shortest thing in the form. These assertions open the tooltip the way a
  // user does rather than reading the prop, so help that renders but never reveals
  // still fails.
  it('explains the keyed form on hover, since it is the only way to free other keys', async () => {
    renderForm(CONDITION_TYPE_VALUE);

    await userEvent.hover(screen.getByRole('img', { name: 'More information about condition filters' }));

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toContain("tags.a IN ('x','y')"));
  });

  it('explains on hover that a target condition never blocks a create', async () => {
    renderForm(CONDITION_TYPE_TARGET);

    await userEvent.hover(screen.getByRole('img', { name: 'More information about condition filters' }));

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toMatch(/never blocks a create/i));
  });

  it('explains on hover that restricting both purposes takes two conditions', async () => {
    // The authoring consequence of one filter per condition, and the only place the UI
    // says it: the type help, not the filter help.
    renderForm(CONDITION_TYPE_VALUE);

    await userEvent.hover(screen.getByRole('img', { name: 'More information about condition types' }));

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toMatch(/two conditions/i));
  });

  it('reveals the hover help on keyboard focus, not only on hover', async () => {
    // The explanation is the only place the keyed form is documented in the UI, so it has
    // to be reachable without a pointer.
    renderForm(CONDITION_TYPE_VALUE);
    const icon = screen.getByRole('img', { name: 'More information about condition filters' });

    expect(icon).toHaveAttribute('tabindex', '0');
    fireEvent.focus(icon);

    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toContain('tags.a IN'));
  });
});

describe('ConditionForm — a partial wildcard scope is refused', () => {
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
    const base = draft({ type: CONDITION_TYPE_VALUE, filter: "tag_key = 'env'", scope: 'scoped' as const });
    expect(isConditionDraftFillable({ ...base, scopePattern: 'team/alpha' })).toBe(true);
    expect(isConditionDraftFillable({ ...base, scopePattern: 'team/*' })).toBe(false);
  });

  it('still fills an all-scoped draft, which is what sends the bare wildcard', () => {
    // The guard must not catch the one wildcard that works: `All ...` sends `*`
    // without going through scopePattern at all.
    expect(
      isConditionDraftFillable(draft({ type: CONDITION_TYPE_VALUE, filter: "tag_key = 'env'", scope: 'all' as const })),
    ).toBe(true);
  });
});
