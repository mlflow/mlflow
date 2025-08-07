import { isFinishedRunStatus, isActiveRunStatus } from './runStatusUtils';
import { FINISHED_RUN_STATUSES } from '../constants';

describe('runStatusUtils', () => {
  describe('isFinishedRunStatus', () => {
    it('returns true for FINISHED status', () => {
      expect(isFinishedRunStatus('FINISHED')).toBe(true);
    });

    it('returns true for FAILED status', () => {
      expect(isFinishedRunStatus('FAILED')).toBe(true);
    });

    it('returns true for KILLED status', () => {
      expect(isFinishedRunStatus('KILLED')).toBe(true);
    });

    it('returns false for RUNNING status', () => {
      expect(isFinishedRunStatus('RUNNING')).toBe(false);
    });

    it('returns false for SCHEDULED status', () => {
      expect(isFinishedRunStatus('SCHEDULED')).toBe(false);
    });

    it('handles all finished statuses from constant array', () => {
      FINISHED_RUN_STATUSES.forEach((status) => {
        expect(isFinishedRunStatus(status as any)).toBe(true);
      });
    });
  });

  describe('isActiveRunStatus', () => {
    it('returns true for RUNNING status', () => {
      expect(isActiveRunStatus('RUNNING')).toBe(true);
    });

    it('returns true for SCHEDULED status', () => {
      expect(isActiveRunStatus('SCHEDULED')).toBe(true);
    });

    it('returns false for FINISHED status', () => {
      expect(isActiveRunStatus('FINISHED')).toBe(false);
    });

    it('returns false for FAILED status', () => {
      expect(isActiveRunStatus('FAILED')).toBe(false);
    });

    it('returns false for KILLED status', () => {
      expect(isActiveRunStatus('KILLED')).toBe(false);
    });

    it('returns true for non-finished statuses', () => {
      const activeStatuses = ['RUNNING', 'SCHEDULED'];
      activeStatuses.forEach((status) => {
        expect(isActiveRunStatus(status as any)).toBe(true);
      });
    });
  });

  describe('status constants', () => {
    it('FINISHED_RUN_STATUSES contains the correct statuses', () => {
      expect(FINISHED_RUN_STATUSES).toEqual(['FINISHED', 'FAILED', 'KILLED']);
    });

    it('covers the expected finished run statuses', () => {
      const expectedFinishedStatuses = ['FINISHED', 'FAILED', 'KILLED'];
      expect(FINISHED_RUN_STATUSES).toEqual(expectedFinishedStatuses);
    });
  });

  describe('edge cases', () => {
    it('isFinishedRunStatus handles undefined gracefully', () => {
      expect(isFinishedRunStatus(undefined as any)).toBe(false);
    });

    it('isActiveRunStatus handles undefined gracefully', () => {
      expect(isActiveRunStatus(undefined as any)).toBe(true); // !isFinishedRunStatus(undefined) = !false = true
    });

    it('isFinishedRunStatus handles null gracefully', () => {
      expect(isFinishedRunStatus(null as any)).toBe(false);
    });

    it('isActiveRunStatus handles null gracefully', () => {
      expect(isActiveRunStatus(null as any)).toBe(true); // !isFinishedRunStatus(null) = !false = true
    });

    it('isFinishedRunStatus handles invalid status gracefully', () => {
      expect(isFinishedRunStatus('INVALID_STATUS' as any)).toBe(false);
    });

    it('isActiveRunStatus handles invalid status gracefully', () => {
      expect(isActiveRunStatus('INVALID_STATUS' as any)).toBe(true); // !isFinishedRunStatus('INVALID_STATUS') = !false = true
    });
  });
});
