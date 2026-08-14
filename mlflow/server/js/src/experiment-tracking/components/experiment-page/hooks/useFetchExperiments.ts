import { useContext } from 'react';

import { GetExperimentsContext } from '../contexts/GetExperimentsContext';

export const useFetchExperiments = () => {
  const getExperimentsContext = useContext(GetExperimentsContext);

  if (!getExperimentsContext) {
    throw new Error('Trying to use GetExperimentsContext actions outside of the context!');
  }

  return getExperimentsContext;
};
