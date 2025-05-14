import React, { ReactNode, useMemo } from 'react';
import { ExperimentPageUIState, createExperimentPageUIState } from '../models/ExperimentPageUIState';

const ExperimentPageUISetStateContext = React.createContext<
  React.Dispatch<React.SetStateAction<ExperimentPageUIState>>
>((state) => state);

// Creates contexts for setting current UI state
export const ExperimentPageUIStateContextProvider = ({
  children,
  setUIState,
}: {
  children: ReactNode;
  setUIState: React.Dispatch<React.SetStateAction<ExperimentPageUIState>>;
}) => (
  <ExperimentPageUISetStateContext.Provider value={setUIState}>{children}</ExperimentPageUISetStateContext.Provider>
);

export const useUpdateExperimentViewUIState = () => React.useContext(ExperimentPageUISetStateContext);
