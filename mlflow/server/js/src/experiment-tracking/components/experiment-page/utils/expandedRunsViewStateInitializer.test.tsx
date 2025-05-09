import { expandedEvaluationRunRowsUIStateInitializer } from './expandedRunsViewStateInitializer';
import { createBaseExperimentEntity, createBaseRunsData, createBaseUIState } from './test-utils';

describe('expandedRunsViewStateInitializer', () => {
  test("it should not change ui state if it's already seeded", () => {
    const initialState = createBaseUIState();
    const baseRunsData = createBaseRunsData();
    const experiments = [createBaseExperimentEntity()];
    const updatedState = expandedEvaluationRunRowsUIStateInitializer(experiments, initialState, baseRunsData, true);

    // Should be unchanged because it's already seeded
    expect(updatedState).toEqual(initialState);
  });
});
