import { renderHook, act } from '@testing-library/react';
import { describe, test, expect } from '@jest/globals';
import {
  ExperimentEvaluationRunsRowVisibilityProvider,
  useExperimentEvaluationRunsRowVisibility,
} from './useExperimentEvaluationRunsRowVisibility';
import { RUNS_VISIBILITY_MODE } from '../../../components/experiment-page/models/ExperimentPageUIState';

describe('useExperimentEvaluationRunsRowVisibility', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <ExperimentEvaluationRunsRowVisibilityProvider>{children}</ExperimentEvaluationRunsRowVisibilityProvider>
  );

  const renderConfiguredHook = () => renderHook(() => useExperimentEvaluationRunsRowVisibility(), { wrapper });

  describe('initial state', () => {
    test('should have SHOWALL mode by default', () => {
      const { result } = renderConfiguredHook();

      expect(result.current.visibilityMode).toBe(RUNS_VISIBILITY_MODE.SHOWALL);
      expect(result.current.usingCustomVisibility).toBe(false);
    });

    test('should show all rows initially', () => {
      const { result } = renderConfiguredHook();

      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(false);
      expect(result.current.isRowHidden('run-2', 5, 'FINISHED')).toBe(false);
      expect(result.current.isRowHidden('run-3', 15, 'FAILED')).toBe(false);
    });
  });

  describe('setVisibilityMode', () => {
    test('should switch to HIDEALL mode', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.HIDEALL);
      });

      expect(result.current.visibilityMode).toBe(RUNS_VISIBILITY_MODE.HIDEALL);
      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(true);
      expect(result.current.isRowHidden('run-2', 15, 'FINISHED')).toBe(true);
    });

    test('should switch to FIRST_10_RUNS mode', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      expect(result.current.visibilityMode).toBe(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(false);
      expect(result.current.isRowHidden('run-2', 9, 'RUNNING')).toBe(false);
      expect(result.current.isRowHidden('run-3', 10, 'RUNNING')).toBe(true);
      expect(result.current.isRowHidden('run-4', 15, 'RUNNING')).toBe(true);
    });

    test('should switch to HIDE_FINISHED_RUNS mode', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS);
      });

      expect(result.current.visibilityMode).toBe(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS);
      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(false);
      expect(result.current.isRowHidden('run-2', 5, 'SCHEDULED')).toBe(false);
      expect(result.current.isRowHidden('run-3', 10, 'FINISHED')).toBe(true);
      expect(result.current.isRowHidden('run-4', 15, 'FAILED')).toBe(true);
      expect(result.current.isRowHidden('run-5', 20, 'KILLED')).toBe(true);
    });

    test('should clear overrides when switching modes', () => {
      const { result } = renderConfiguredHook();

      // First, set FIRST_10_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Toggle a visible row (row 5) to hide it
      act(() => {
        result.current.toggleRowVisibility('run-5', 5, 'RUNNING');
      });

      expect(result.current.usingCustomVisibility).toBe(true);
      expect(result.current.isRowHidden('run-5', 5, 'RUNNING')).toBe(true);

      // Switch to SHOWALL - should clear overrides
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.SHOWALL);
      });

      expect(result.current.usingCustomVisibility).toBe(false);
      expect(result.current.isRowHidden('run-5', 5, 'RUNNING')).toBe(false);
    });
  });

  describe('toggleRowVisibility - override pattern', () => {
    test('should hide a visible row in SHOWALL mode', () => {
      const { result } = renderConfiguredHook();

      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(false);
      expect(result.current.usingCustomVisibility).toBe(false);

      act(() => {
        result.current.toggleRowVisibility('run-1', 0, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(true);
      expect(result.current.usingCustomVisibility).toBe(true);
    });

    test('should show a hidden row after toggling twice', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.toggleRowVisibility('run-1', 0, 'RUNNING');
      });
      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(true);

      act(() => {
        result.current.toggleRowVisibility('run-1', 0, 'RUNNING');
      });
      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(false);
      expect(result.current.usingCustomVisibility).toBe(false);
    });

    test('should show a hidden row in FIRST_10_RUNS mode when toggled (bug fix)', () => {
      const { result } = renderConfiguredHook();

      // Set FIRST_10_RUNS mode - rows 10+ are hidden by default
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Row 15 should be hidden by the mode
      expect(result.current.isRowHidden('run-15', 15, 'RUNNING')).toBe(true);

      // Toggle row 15 - should SHOW it (override the mode)
      act(() => {
        result.current.toggleRowVisibility('run-15', 15, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-15', 15, 'RUNNING')).toBe(false);
      expect(result.current.usingCustomVisibility).toBe(true);
    });

    test('should hide a visible row in FIRST_10_RUNS mode when toggled', () => {
      const { result } = renderConfiguredHook();

      // Set FIRST_10_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Row 5 should be visible by the mode
      expect(result.current.isRowHidden('run-5', 5, 'RUNNING')).toBe(false);

      // Toggle row 5 - should HIDE it (override the mode)
      act(() => {
        result.current.toggleRowVisibility('run-5', 5, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-5', 5, 'RUNNING')).toBe(true);
      expect(result.current.usingCustomVisibility).toBe(true);
    });

    test('should preserve other hidden rows when toggling in FIRST_10_RUNS mode', () => {
      const { result } = renderConfiguredHook();

      // Set FIRST_10_RUNS mode - rows 10+ are hidden
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Toggle row 5 to hide it
      act(() => {
        result.current.toggleRowVisibility('run-5', 5, 'RUNNING');
      });

      // Verify row 5 is hidden
      expect(result.current.isRowHidden('run-5', 5, 'RUNNING')).toBe(true);

      // Verify rows 10+ are still hidden by the mode
      expect(result.current.isRowHidden('run-10', 10, 'RUNNING')).toBe(true);
      expect(result.current.isRowHidden('run-15', 15, 'RUNNING')).toBe(true);
      expect(result.current.isRowHidden('run-20', 20, 'RUNNING')).toBe(true);
    });

    test('should handle multiple toggles independently', () => {
      const { result } = renderConfiguredHook();

      // Set FIRST_10_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Toggle multiple rows
      act(() => {
        result.current.toggleRowVisibility('run-5', 5, 'RUNNING'); // visible -> hidden
        result.current.toggleRowVisibility('run-15', 15, 'RUNNING'); // hidden -> visible
        result.current.toggleRowVisibility('run-20', 20, 'RUNNING'); // hidden -> visible
      });

      expect(result.current.isRowHidden('run-5', 5, 'RUNNING')).toBe(true); // overridden to hide
      expect(result.current.isRowHidden('run-9', 9, 'RUNNING')).toBe(false); // still visible
      expect(result.current.isRowHidden('run-10', 10, 'RUNNING')).toBe(true); // still hidden by mode
      expect(result.current.isRowHidden('run-15', 15, 'RUNNING')).toBe(false); // overridden to show
      expect(result.current.isRowHidden('run-20', 20, 'RUNNING')).toBe(false); // overridden to show
    });

    test('should work with HIDEALL mode', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.HIDEALL);
      });

      // All rows hidden by mode
      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(true);

      // Toggle to show it
      act(() => {
        result.current.toggleRowVisibility('run-1', 0, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-1', 0, 'RUNNING')).toBe(false);
      expect(result.current.usingCustomVisibility).toBe(true);
    });
  });

  describe('explicit visibility - status/index changes', () => {
    test('should keep run hidden when user hides it and status later changes', () => {
      const { result } = renderConfiguredHook();

      // Set HIDE_FINISHED_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS);
      });

      // Run is RUNNING (visible by mode)
      expect(result.current.isRowHidden('run-1', 5, 'RUNNING')).toBe(false);

      // User explicitly hides it
      act(() => {
        result.current.toggleRowVisibility('run-1', 5, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-1', 5, 'RUNNING')).toBe(true);

      // Status changes to FINISHED (mode would hide it now)
      // But user's explicit choice should persist - still hidden
      expect(result.current.isRowHidden('run-1', 5, 'FINISHED')).toBe(true);
    });

    test('should keep run visible when user shows it and status later changes', () => {
      const { result } = renderConfiguredHook();

      // Set HIDE_FINISHED_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS);
      });

      // Run is FINISHED (hidden by mode)
      expect(result.current.isRowHidden('run-1', 5, 'FINISHED')).toBe(true);

      // User explicitly shows it
      act(() => {
        result.current.toggleRowVisibility('run-1', 5, 'FINISHED');
      });

      expect(result.current.isRowHidden('run-1', 5, 'FINISHED')).toBe(false);

      // Status changes to RUNNING (mode would show it now)
      // But user's explicit choice should persist - still visible
      expect(result.current.isRowHidden('run-1', 5, 'RUNNING')).toBe(false);
    });

    test('should keep run hidden when user hides it and index later changes', () => {
      const { result } = renderConfiguredHook();

      // Set FIRST_10_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Run at index 5 (visible by mode)
      expect(result.current.isRowHidden('run-1', 5, 'RUNNING')).toBe(false);

      // User explicitly hides it
      act(() => {
        result.current.toggleRowVisibility('run-1', 5, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-1', 5, 'RUNNING')).toBe(true);

      // Index changes to 15 due to sorting/filtering (mode would hide it now)
      // But user's explicit choice should persist - still hidden
      expect(result.current.isRowHidden('run-1', 15, 'RUNNING')).toBe(true);
    });

    test('should keep run visible when user shows it and index later changes', () => {
      const { result } = renderConfiguredHook();

      // Set FIRST_10_RUNS mode
      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      // Run at index 15 (hidden by mode)
      expect(result.current.isRowHidden('run-1', 15, 'RUNNING')).toBe(true);

      // User explicitly shows it
      act(() => {
        result.current.toggleRowVisibility('run-1', 15, 'RUNNING');
      });

      expect(result.current.isRowHidden('run-1', 15, 'RUNNING')).toBe(false);

      // Index changes to 5 due to sorting/filtering (mode would show it now)
      // But user's explicit choice should persist - still visible
      expect(result.current.isRowHidden('run-1', 5, 'RUNNING')).toBe(false);
    });
  });

  describe('usingCustomVisibility flag', () => {
    test('should be false when no overrides exist', () => {
      const { result } = renderConfiguredHook();

      expect(result.current.usingCustomVisibility).toBe(false);

      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
      });

      expect(result.current.usingCustomVisibility).toBe(false);
    });

    test('should be true when overrides exist', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.setVisibilityMode(RUNS_VISIBILITY_MODE.FIRST_10_RUNS);
        result.current.toggleRowVisibility('run-5', 5, 'RUNNING');
      });

      expect(result.current.usingCustomVisibility).toBe(true);
    });

    test('should become false when all overrides are removed', () => {
      const { result } = renderConfiguredHook();

      act(() => {
        result.current.toggleRowVisibility('run-1', 0, 'RUNNING');
        result.current.toggleRowVisibility('run-2', 1, 'RUNNING');
      });

      expect(result.current.usingCustomVisibility).toBe(true);

      // Remove all overrides
      act(() => {
        result.current.toggleRowVisibility('run-1', 0, 'RUNNING');
        result.current.toggleRowVisibility('run-2', 1, 'RUNNING');
      });

      expect(result.current.usingCustomVisibility).toBe(false);
    });
  });
});
