import { describe, it, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { useCanEditExperimentCustomViews } from './useCanEditExperimentCustomViews';

// OSS has no per-experiment write-permission system for tags, so this hook is a
// stub that always grants edit access (see the hook's own docstring for the
// Databricks-parity rationale). This test only pins that stub behavior; the
// interface exists so it stays a one-line swap if OSS later adds a real ACL check.
describe('useCanEditExperimentCustomViews', () => {
  it('always reports canEdit true and not loading, regardless of the experiment id', () => {
    const withId = renderHook(() => useCanEditExperimentCustomViews('exp-123'));
    expect(withId.result.current).toEqual({ canEdit: true, isLoading: false });

    const withoutId = renderHook(() => useCanEditExperimentCustomViews(undefined));
    expect(withoutId.result.current).toEqual({ canEdit: true, isLoading: false });
  });
});
