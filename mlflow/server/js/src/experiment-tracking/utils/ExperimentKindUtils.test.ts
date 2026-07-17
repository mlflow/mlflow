import { describe, test, expect } from '@jest/globals';
import { ExperimentKind } from '../constants';
import { WorkflowType } from '../../common/contexts/WorkflowTypeContext';
import {
  getWorkflowTypeForExperimentKind,
  getExperimentKindFromTags,
  EXPERIMENT_KIND_TAG_KEY,
} from './ExperimentKindUtils';

describe('getWorkflowTypeForExperimentKind', () => {
  test('maps GenAI kinds to GENAI', () => {
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.GENAI_DEVELOPMENT)).toBe(WorkflowType.GENAI);
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.GENAI_DEVELOPMENT_INFERRED)).toBe(WorkflowType.GENAI);
  });

  test('maps classic-ML kinds to MACHINE_LEARNING', () => {
    for (const kind of [
      ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
      ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED,
      ExperimentKind.FINETUNING,
      ExperimentKind.REGRESSION,
      ExperimentKind.CLASSIFICATION,
      ExperimentKind.FORECASTING,
      ExperimentKind.AUTOML,
    ]) {
      expect(getWorkflowTypeForExperimentKind(kind)).toBe(WorkflowType.MACHINE_LEARNING);
    }
  });

  test('returns undefined for ambiguous / absent kinds so callers keep the default', () => {
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.NO_INFERRED_TYPE)).toBeUndefined();
    expect(getWorkflowTypeForExperimentKind(ExperimentKind.EMPTY)).toBeUndefined();
    expect(getWorkflowTypeForExperimentKind(undefined)).toBeUndefined();
  });
});

describe('getExperimentKindFromTags', () => {
  test('reads the experiment kind tag', () => {
    expect(getExperimentKindFromTags([{ key: EXPERIMENT_KIND_TAG_KEY, value: 'custom_model_development' }])).toBe(
      ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
    );
  });

  test('returns undefined when the tag is missing', () => {
    expect(getExperimentKindFromTags([{ key: 'other', value: 'x' }])).toBeUndefined();
    expect(getExperimentKindFromTags([])).toBeUndefined();
    expect(getExperimentKindFromTags(undefined)).toBeUndefined();
  });
});
