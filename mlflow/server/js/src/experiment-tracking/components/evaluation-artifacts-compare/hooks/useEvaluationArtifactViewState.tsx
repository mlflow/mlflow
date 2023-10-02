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

  useEffect(
    () =>
      updateViewState({
        artifactViewState: {
          selectedTables,
          groupByCols,
          outputColumn,
        },
      }),
    [updateViewState, selectedTables, groupByCols, outputColumn],
  );

  return {
    selectedTables,
    groupByCols,
    outputColumn,
    setSelectedTables,
    setGroupByCols,
    setOutputColumn,
  };
};
