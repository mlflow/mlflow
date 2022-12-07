import { useContext } from 'react';

import { GetExperimentRunsContext } from '../contexts/GetExperimentRunsContext';

export const useFetchExperimentRuns = () => {
  const getExperimentRunsContextValue = useContext(GetExperimentRunsContext);

  if (!getExperimentRunsContextValue) {
    throw new Error('Trying to use SearchExperimentRunsContext actions outside of the context!');
  }

  return getExperimentRunsContextValue;
};
