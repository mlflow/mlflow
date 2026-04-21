import React, { createContext, useCallback, useContext, useMemo } from 'react';
import { useLocalStorage } from '../../shared/web-shared/hooks/useLocalStorage';
import { useNavigate, useParams, useSearchParams } from '../utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';

export enum WorkflowType {
  GENAI = 'genai',
  MACHINE_LEARNING = 'machine_learning',
}

/**
 * Reads the workflowType search param from the URL. Returns the param value
 * if it is a valid WorkflowType, otherwise returns the provided fallback.
 */
export const getWorkflowTypeFromUrl = (searchParams: URLSearchParams, fallback: WorkflowType): WorkflowType => {
  const value = searchParams.get('workflowType');
  if (value === WorkflowType.GENAI || value === WorkflowType.MACHINE_LEARNING) {
    return value;
  }
  return fallback;
};

/**
 * Builds a query string containing the workflowType and any existing workspace
 * param. Used when navigating between experiment pages to preserve sidebar state.
 */
export const buildWorkflowTypeQueryString = (
  workflowType: WorkflowType,
  currentSearchParams: URLSearchParams,
): string => {
  const params = new URLSearchParams();
  params.set('workflowType', workflowType);
  const workspace = currentSearchParams.get('workspace');
  if (workspace) {
    params.set('workspace', workspace);
  }
  return params.toString();
};

const WORKFLOW_TYPE_STORAGE_KEY = 'mlflow.workflowType';
const WORKFLOW_TYPE_STORAGE_VERSION = 1;
const DEFAULT_WORKFLOW_TYPE = WorkflowType.GENAI;

interface WorkflowTypeContextType {
  workflowType: WorkflowType;
  setWorkflowType: (workflowType: WorkflowType) => void;
}

const WorkflowTypeContext = createContext<WorkflowTypeContextType>({
  workflowType: DEFAULT_WORKFLOW_TYPE,
  setWorkflowType: () => {},
});

export const WorkflowTypeProvider = ({ children }: { children: React.ReactNode }) => {
  const navigate = useNavigate();
  const [workflowType, setWorkflowType] = useLocalStorage<WorkflowType>({
    key: WORKFLOW_TYPE_STORAGE_KEY,
    version: WORKFLOW_TYPE_STORAGE_VERSION,
    initialValue: DEFAULT_WORKFLOW_TYPE,
  });

  const { experimentId } = useParams();
  const [searchParams] = useSearchParams();
  const effectiveWorkflowType = getWorkflowTypeFromUrl(searchParams, workflowType);

  const handleWorkflowTypeChange = useCallback(
    (newWorkflowType: WorkflowType) => {
      if (newWorkflowType === effectiveWorkflowType) {
        return;
      }

      setWorkflowType(newWorkflowType);
      if (experimentId) {
        const query = buildWorkflowTypeQueryString(newWorkflowType, searchParams);
        navigate(`${Routes.getExperimentPageRoute(experimentId)}?${query}`);
      }
    },
    [effectiveWorkflowType, experimentId, navigate, searchParams, setWorkflowType],
  );

  const contextValue = useMemo(
    () => ({
      workflowType: effectiveWorkflowType,
      setWorkflowType: handleWorkflowTypeChange,
    }),
    [effectiveWorkflowType, handleWorkflowTypeChange],
  );

  return <WorkflowTypeContext.Provider value={contextValue}>{children}</WorkflowTypeContext.Provider>;
};

export const useWorkflowType = (): WorkflowTypeContextType => {
  return useContext(WorkflowTypeContext);
};
