import { create } from '@databricks/web-shared/zustand';

import type { Assessment } from '../ModelTrace.types';

type TraceActionCacheAction = { action: 'ADD' | 'DELETE'; assessment?: Assessment };

export const useTraceCachedActions = create<{
  assessmentActions: Record<string, TraceActionCacheAction[]>;
  /**
   * Log an added or updated assessment
   */
  logAddedAssessment: (traceId: string, assessment: Assessment) => void;
  /**
   * Log a removed assessment
   */
  logRemovedAssessment: (traceId: string, assessment: Assessment) => void;
  /**
   * Reconstruct the list of assessments by applying cached actions on top of the initial list
   */
  reconstructAssessments: (initialAssessments: Assessment[], actions: TraceActionCacheAction[]) => Assessment[];
  resetCache: () => void;
}>((set) => ({
  assessmentActions: {},
  resetCache: () => set(() => ({ assessmentActions: {} })),
  logAddedAssessment: (traceId: string, assessment?: Assessment) =>
    set((state) => {
      if (!assessment) {
        return state;
      }
      return {
        assessmentActions: {
          ...state.assessmentActions,
          [traceId]: [...(state.assessmentActions[traceId] || []), { action: 'ADD', assessment }],
        },
      };
    }),
  logRemovedAssessment: (traceId: string, assessment: Assessment) =>
    set((state) => {
      if (!assessment) {
        return state;
      }
      return {
        assessmentActions: {
          ...state.assessmentActions,
          [traceId]: [...(state.assessmentActions[traceId] || []), { action: 'DELETE', assessment }],
        },
      };
    }),
  reconstructAssessments: (initialAssessments: Assessment[], actions?: TraceActionCacheAction[]) => {
    if (!actions) {
      return initialAssessments;
    }
    let assessments = [...initialAssessments];
    actions.forEach(({ action, assessment }) => {
      if (action === 'ADD' && assessment) {
        assessments.unshift(assessment);
      } else if (action === 'DELETE' && assessment) {
        assessments = assessments.filter((a) => a.assessment_id !== assessment.assessment_id);
      }
    });
    return assessments;
  },
}));
