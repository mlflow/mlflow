import { ModelTraceExplorerPreferencesProvider as LegacyModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerPreferencesProvider as RedesignedModelTraceExplorerPreferencesProvider } from './v2/ModelTraceExplorerPreferencesContext';
import { shouldEnableRedesignedTraceExplorer } from './shouldEnableRedesignedTraceExplorer';

export interface ModelTraceExplorerPreferencesProviderProps {
  children: React.ReactNode;
}

export const ModelTraceExplorerPreferencesProvider = ({
  children,
}: ModelTraceExplorerPreferencesProviderProps): JSX.Element => {
  const Provider = shouldEnableRedesignedTraceExplorer()
    ? RedesignedModelTraceExplorerPreferencesProvider
    : LegacyModelTraceExplorerPreferencesProvider;
  return <Provider>{children}</Provider>;
};
