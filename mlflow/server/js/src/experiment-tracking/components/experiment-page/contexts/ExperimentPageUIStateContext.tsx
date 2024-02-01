import React, { ReactNode, useMemo } from 'react';
import { ExperimentPageUIStateV2, createExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';

const ExperimentPageUISetStateContext = React.createContext<
  React.Dispatch<React.SetStateAction<ExperimentPageUIStateV2>>
>((state) => state);

// Creates contexts for setting current UI state
export const ExperimentPageUIStateContextProvider = ({
  children,
  setUIState,
}: {
  children: ReactNode;
  setUIState: React.Dispatch<React.SetStateAction<ExperimentPageUIStateV2>>;
}) => (
  <ExperimentPageUISetStateContext.Provider value={setUIState}>{children}</ExperimentPageUISetStateContext.Provider>
);

export const useUpdateExperimentViewUIState = () => React.useContext(ExperimentPageUISetStateContext);
