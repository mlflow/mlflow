import { describe, it, expect } from '@jest/globals';

import {
  draftToStagedCondition,
  isMutationConditionDraftDirty,
  isMutationConditionDraftFillable,
  MUTATION_CONDITION_DRAFT_DEFAULT,
  type MutationConditionDraft,
} from './MutationConditionForm';

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
