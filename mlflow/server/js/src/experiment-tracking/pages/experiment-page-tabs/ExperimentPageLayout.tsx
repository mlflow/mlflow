import { createContext, useContext } from 'react';

// The experiment page's layout element. With the new eval runs UI, the traces drawer is scoped to it
// while the MLflow Assistant is open, so the drawer fills the page beside the Assistant panel.
const ExperimentPageLayoutContext = createContext<HTMLElement | null>(null);

export const useExperimentPageLayoutContainer = () => useContext(ExperimentPageLayoutContext);
export const ExperimentPageLayoutProvider = ExperimentPageLayoutContext.Provider;
