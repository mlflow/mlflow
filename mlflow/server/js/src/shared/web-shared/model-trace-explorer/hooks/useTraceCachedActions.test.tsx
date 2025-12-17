import { describe, beforeEach, it, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';

import { useTraceCachedActions } from './useTraceCachedActions';
import type { Assessment } from '../ModelTrace.types';

describe('useTraceCachedActions', () => {
  beforeEach(() => {
    const { result } = renderHook(() => useTraceCachedActions());
    act(() => {
      result.current.resetCache();
    });
  });

  const mockAssessment: Assessment = {
    assessment_id: 'test-id-1',
    name: 'Test Assessment',
    trace_id: 'trace-1',
  } as any;

  describe('reconstructAssessments', () => {
    it('should apply multiple add and delete operations in sequence', () => {
      const { result } = renderHook(() => useTraceCachedActions());
      const assessment2: Assessment = {
        assessment_id: 'test-id-2',
        name: 'Second Assessment',
        trace_id: 'trace-1',
      } as any;
      const initial: Assessment[] = [mockAssessment, { assessment_id: 'existing-1' } as any];

      act(() => {
        result.current.logAddedAssessment('trace-1', assessment2);
        result.current.logAddedAssessment('trace-1', { assessment_id: 'test-id-3' } as any);
        result.current.logRemovedAssessment('trace-1', mockAssessment);
      });

      const reconstructed = result.current.reconstructAssessments(initial, result.current.assessmentActions['trace-1']);

      expect(reconstructed).toHaveLength(3);
      expect(reconstructed[0].assessment_id).toBe('test-id-3');
      expect(reconstructed[1].assessment_id).toBe('test-id-2');
      expect(reconstructed[2].assessment_id).toBe('existing-1');
    });
  });
});
