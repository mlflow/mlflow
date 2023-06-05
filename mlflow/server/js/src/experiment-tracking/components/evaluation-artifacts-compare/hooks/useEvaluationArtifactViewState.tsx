import { useEffect, useState } from 'react';
import { SearchExperimentRunsViewState } from '../../experiment-page/models/SearchExperimentRunsViewState';
import { UpdateExperimentViewStateFn } from '../../../types';

export const useEvaluationArtifactViewState = (
  viewState: SearchExperimentRunsViewState,
  updateViewState: UpdateExperimentViewStateFn,
) => {
  const { artifactViewState = {} } = viewState;

  const [selectedTables, setSelectedTables] = useState<string[]>(
    artifactViewState.selectedTables || [],
  );
  const [groupByCols, setGroupByCols] = useState<string[]>(artifactViewState.groupByCols || []);
  const [outputColumn, setOutputColumn] = useState(artifactViewState.outputColumn || '');
  const [intersectingOnly, setIntersectingOnly] = useState(
    artifactViewState.intersectingOnly || false,
  );

  useEffect(
    () =>
      updateViewState({
        artifactViewState: {
          selectedTables,
          groupByCols,
          outputColumn,
          intersectingOnly,
        },
      }),
    [updateViewState, selectedTables, groupByCols, outputColumn, intersectingOnly],
  );

  return {
    selectedTables,
    groupByCols,
    outputColumn,
    intersectingOnly,
    setSelectedTables,
    setGroupByCols,
    setOutputColumn,
    setIntersectingOnly,
  };
};
