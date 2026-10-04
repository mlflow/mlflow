import { describe, it, expect } from '@jest/globals';

import {
  draftToAddRequest,
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
    const scoped = draft({ scope: 'parent', resourceType: 'run', targetCondition: "tags.x = 'y'" });
    expect(isMutationConditionDraftFillable(scoped)).toBe(false);
    expect(isMutationConditionDraftFillable({ ...scoped, parentResourceId: '42' })).toBe(true);
  });

  it('sends an absent filter as null, never as an empty string', () => {
    // ``''`` would reach the condition parser and be reported as a filter syntax
    // error rather than as an absent filter.
    const request = draftToAddRequest(draft({ targetCondition: "tags.x = 'y'" }), 7);
    expect(request.value_condition).toBeNull();
    expect(request.target_condition).toBe("tags.x = 'y'");
  });

  it('sends both parent fields together, or neither', () => {
    // The server enforces the pair with a CHECK constraint, so half a pair is a
    // rejected write.
    const unscoped = draftToAddRequest(draft({ resourceType: 'run', targetCondition: "tags.x = 'y'" }), 7);
    expect(unscoped.parent_resource_type).toBeNull();
    expect(unscoped.parent_resource_id).toBeNull();

    const scoped = draftToAddRequest(
      draft({ resourceType: 'run', scope: 'parent', parentResourceId: '42', targetCondition: "tags.x = 'y'" }),
      7,
    );
    expect(scoped.parent_resource_type).toBe('experiment');
    expect(scoped.parent_resource_id).toBe('42');
  });

  it('derives the parent type from the resource type rather than trusting the draft', () => {
    // The draft never carries a parent type. If it did, it could disagree with the
    // resource type and persist a scope that can never match.
    const versionScoped = draftToAddRequest(
      draft({
        resourceType: 'registered_model_version',
        scope: 'parent',
        parentResourceId: 'my-model',
        targetCondition: "tags.x = 'y'",
      }),
      7,
    );
    expect(versionScoped.parent_resource_type).toBe('registered_model');
  });

  it('drops a parent scope the resource type cannot have', () => {
    // ``experiment`` is parentless. A scope left over from a previous selection
    // must not travel, or the server rejects the write for a field the admin can
    // no longer see.
    const request = draftToAddRequest(
      draft({ resourceType: 'experiment', scope: 'parent', parentResourceId: 'stale', targetCondition: "tags.x = 'y'" }),
      7,
    );
    expect(request.parent_resource_type).toBeNull();
    expect(request.parent_resource_id).toBeNull();
  });

  it('trims the filters it sends', () => {
    const request = draftToAddRequest(draft({ targetCondition: "  tags.x = 'y'  " }), 7);
    expect(request.target_condition).toBe("tags.x = 'y'");
  });

  it('reports a default draft as clean and any edit as dirty', () => {
    expect(isMutationConditionDraftDirty(draft())).toBe(false);
    expect(isMutationConditionDraftDirty(draft({ targetCondition: 'x' }))).toBe(true);
    expect(isMutationConditionDraftDirty(draft({ resourceType: 'run' }))).toBe(true);
  });
});
