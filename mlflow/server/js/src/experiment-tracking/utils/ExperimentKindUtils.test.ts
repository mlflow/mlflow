import { describe, test, expect } from '@jest/globals';
import { ExperimentKind } from '../constants';
import { WorkflowType } from '../../common/contexts/WorkflowTypeContext';
import { getWorkflowTypeForExperimentKind } from './ExperimentKindUtils';

describe('getWorkflowTypeForExperimentKind', () => {
  test('maps GenAI kinds to GENAI', () => {
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.GENAI_DEVELOPMENT)).toBe(WorkflowType.GENAI);
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.GENAI_DEVELOPMENT_INFERRED)).toBe(WorkflowType.GENAI);
  });

  test('maps ML kinds to MACHINE_LEARNING', () => {
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT)).toBe(
      WorkflowType.MACHINE_LEARNING,
    );
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED)).toBe(
      WorkflowType.MACHINE_LEARNING,
    );
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.FINETUNING)).toBe(WorkflowType.MACHINE_LEARNING);
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.REGRESSION)).toBe(WorkflowType.MACHINE_LEARNING);
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.CLASSIFICATION)).toBe(WorkflowType.MACHINE_LEARNING);
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.FORECASTING)).toBe(WorkflowType.MACHINE_LEARNING);
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.AUTOML)).toBe(WorkflowType.MACHINE_LEARNING);
  });

  test('returns undefined for ambiguous or absent kinds', () => {
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.NO_INFERRED_TYPE)).toBeUndefined();
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.EMPTY)).toBeUndefined();
    expect(getWorkflowTypeForExperimentKind(undefined)).toBeUndefined();
  });
});
