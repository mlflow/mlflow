import { renderHook } from '@testing-library/react';
import {
  isLoggedModelDataGroupDataRow,
  LoggedModelsTableGroupByMode,
  LoggedModelsTableSpecialRowID,
  useLoggedModelTableDataRows,
} from './ExperimentLoggedModelListPageTable.utils';
import type { LoggedModelDataWithSourceRun } from './ExperimentLoggedModelListPageTable.utils';
import type { RunEntity } from '../../types';

describe('ExperimentLoggedModelListPageTable.utils', () => {
  describe('useLoggedModelTableDataRows', () => {
    // Sample data for testing
    const createSampleRun = (runId: string): RunEntity => ({
      info: {
        runName: `Run ${runId}`,
        runUuid: runId,
        experimentId: 'exp-1',
        status: 'FINISHED',
        startTime: 1623456789,
        endTime: 1623456999,
        artifactUri: `artifacts/${runId}`,
        lifecycleStage: 'active',
      },
      data: {
        params: [],
        tags: [],
        metrics: [],
      },
    });

    const createSampleLoggedModel = (
      modelId: string,
      runId?: string,
      sourceRun?: RunEntity,
    ): LoggedModelDataWithSourceRun => ({
      info: {
        model_id: modelId,
        name: `model-${modelId}`,
        source_run_id: runId,
      },
      sourceRun,
    });

    test('should return undefined when no logged models are provided', () => {
      const { result } = renderHook(() =>
        useLoggedModelTableDataRows({
          loggedModelsWithSourceRuns: undefined,
          expandedGroups: [],
        }),
      );

      expect(result.current).toBeUndefined();
    });

    test('should return empty array when logged models array is empty', () => {
      const { result } = renderHook(() =>
        useLoggedModelTableDataRows({
          loggedModelsWithSourceRuns: [],
          expandedGroups: [],
        }),
      );

      expect(result.current).toEqual([]);
    });

    test('should return models as-is when no grouping is specified', () => {
      const run1 = createSampleRun('run-1');
      const run2 = createSampleRun('run-2');

      const model1 = createSampleLoggedModel('model-1', 'run-1', run1);
      const model2 = createSampleLoggedModel('model-2', 'run-2', run2);
      const model3 = createSampleLoggedModel('model-3', 'run-1', run1);

      const loggedModels = [model1, model2, model3];

      const { result } = renderHook(() =>
        useLoggedModelTableDataRows({
          loggedModelsWithSourceRuns: loggedModels,
          expandedGroups: [],
        }),
      );

      // Should return all models without grouping
      expect(result.current).toHaveLength(3);
      expect((result.current?.[0] as LoggedModelDataWithSourceRun).info?.model_id).toBe('model-1');
      expect((result.current?.[1] as LoggedModelDataWithSourceRun).info?.model_id).toBe('model-2');
      expect((result.current?.[2] as LoggedModelDataWithSourceRun).info?.model_id).toBe('model-3');
    });

    test('should group models by run when grouping by runs', () => {
      const run1 = createSampleRun('run-1');
      const run2 = createSampleRun('run-2');

      const model1 = createSampleLoggedModel('model-1', 'run-1', run1);
      const model2 = createSampleLoggedModel('model-2', 'run-2', run2);
      const model3 = createSampleLoggedModel('model-3', 'run-1', run1);
      const model4 = createSampleLoggedModel('model-4', undefined, undefined); // No run ID

      const loggedModels = [model1, model2, model3, model4];

      const { result } = renderHook(() =>
        useLoggedModelTableDataRows({
          loggedModelsWithSourceRuns: loggedModels,
          groupModelsBy: LoggedModelsTableGroupByMode.RUNS,
          expandedGroups: [],
        }),
      );

      // Should return groups but not the models since no groups are expanded
      expect(result.current).toHaveLength(3); // 3 groups: run-1, run-2, and REMAINING_MODELS_GROUP

      const [firstGroup, secondGroup, remainingGroup] = result.current || [];

      // Check that all returned items are groups
      expect(isLoggedModelDataGroupDataRow(firstGroup)).toBe(true);
      expect(isLoggedModelDataGroupDataRow(secondGroup)).toBe(true);
      expect(isLoggedModelDataGroupDataRow(remainingGroup)).toBe(true);

      // Check group IDs - the REMAINING_MODELS_GROUP should be last
      expect(isLoggedModelDataGroupDataRow(firstGroup) && firstGroup.groupUuid).toBe('run-1');
      expect(isLoggedModelDataGroupDataRow(secondGroup) && secondGroup.groupUuid).toBe('run-2');
      expect(isLoggedModelDataGroupDataRow(remainingGroup) && remainingGroup.groupUuid).toBe(
        LoggedModelsTableSpecialRowID.REMAINING_MODELS_GROUP,
      );

      // Check that the source run is correctly set for the groups
      expect(isLoggedModelDataGroupDataRow(firstGroup) && firstGroup.groupData?.sourceRun).toBe(run1);
      expect(isLoggedModelDataGroupDataRow(secondGroup) && secondGroup.groupData?.sourceRun).toBe(run2);
    });

    test('should expand groups that are in the expandedGroups array', () => {
      const run1 = createSampleRun('run-1');
      const run2 = createSampleRun('run-2');

      const model1 = createSampleLoggedModel('model-1', 'run-1', run1);
      const model2 = createSampleLoggedModel('model-2', 'run-2', run2);
      const model3 = createSampleLoggedModel('model-3', 'run-1', run1);
      const model4 = createSampleLoggedModel('model-4', undefined, undefined); // No run ID

      const loggedModels = [model1, model2, model3, model4];

      const { result } = renderHook(() =>
        useLoggedModelTableDataRows({
          loggedModelsWithSourceRuns: loggedModels,
          groupModelsBy: LoggedModelsTableGroupByMode.RUNS,
          expandedGroups: ['run-1', LoggedModelsTableSpecialRowID.REMAINING_MODELS_GROUP],
        }),
      );

      // Should have 3 groups + 2 models from run-1 + 1 model from REMAINING_MODELS_GROUP
      expect(result.current).toHaveLength(6);

      const [firstGroup, model1FromRun1, model2FromRun1, secondGroup, remainingGroup, modelFromRemaining] =
        result.current || [];

      // Check the structure: group -> models -> group -> models -> group
      expect(isLoggedModelDataGroupDataRow(firstGroup) && firstGroup.isGroup).toBe(true);
      expect(isLoggedModelDataGroupDataRow(firstGroup) && firstGroup.groupUuid).toBe('run-1');

      expect(isLoggedModelDataGroupDataRow(secondGroup) && secondGroup.isGroup).toBe(true);
      expect(isLoggedModelDataGroupDataRow(secondGroup) && secondGroup.groupUuid).toBe('run-2');

      // Models from run-1 (expanded)
      expect(model1FromRun1 && 'info' in model1FromRun1).toBe(true);
      expect(model1FromRun1 && !isLoggedModelDataGroupDataRow(model1FromRun1) && model1FromRun1.info?.model_id).toBe(
        'model-1',
      );
      expect(model2FromRun1 && !isLoggedModelDataGroupDataRow(model2FromRun1) && 'info' in model2FromRun1).toBe(true);
      expect(model2FromRun1 && !isLoggedModelDataGroupDataRow(model2FromRun1) && model2FromRun1.info?.model_id).toBe(
        'model-3',
      );

      expect(isLoggedModelDataGroupDataRow(remainingGroup) && remainingGroup.isGroup).toBe(true);
      expect(isLoggedModelDataGroupDataRow(remainingGroup) && remainingGroup.groupUuid).toBe(
        LoggedModelsTableSpecialRowID.REMAINING_MODELS_GROUP,
      );

      // Model from REMAINING_MODELS_GROUP (expanded)
      expect(
        modelFromRemaining && !isLoggedModelDataGroupDataRow(modelFromRemaining) && 'info' in modelFromRemaining,
      ).toBe(true);
      expect(
        modelFromRemaining && !isLoggedModelDataGroupDataRow(modelFromRemaining) && modelFromRemaining.info?.model_id,
      ).toBe('model-4');
    });

    test('should handle empty expandedGroups array', () => {
      const run1 = createSampleRun('run-1');
      const model1 = createSampleLoggedModel('model-1', 'run-1', run1);
      const model2 = createSampleLoggedModel('model-2', 'run-1', run1);

      const loggedModels = [model1, model2];

      const { result } = renderHook(() =>
        useLoggedModelTableDataRows({
          loggedModelsWithSourceRuns: loggedModels,
          groupModelsBy: LoggedModelsTableGroupByMode.RUNS,
          expandedGroups: [],
        }),
      );

      const [group] = result.current || [];

      // Should only have the group, no models
      expect(result.current).toHaveLength(1);
      expect(isLoggedModelDataGroupDataRow(group) && group.isGroup).toBe(true);
      expect(isLoggedModelDataGroupDataRow(group) && group.groupUuid).toBe('run-1');
    });
  });
});
