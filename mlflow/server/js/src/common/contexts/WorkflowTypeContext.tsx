import React, { createContext, useCallback, useContext, useMemo } from 'react';
import { useLocalStorage } from '../../shared/web-shared/hooks/useLocalStorage';
import { useNavigate, useParams } from '../utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';

export enum WorkflowType {
  GENAI = 'genai',
  MACHINE_LEARNING = 'machine_learning',
}

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
  const handleWorkflowTypeChange = useCallback(
    (newWorkflowType: WorkflowType) => {
      if (newWorkflowType === workflowType) {
        return;
      }

      setWorkflowType(newWorkflowType);
      if (experimentId) {
        // if an experiment is active, redirect to experiment default tab on change
        navigate(Routes.getExperimentPageRoute(experimentId));
      }
    },
    [experimentId, navigate, setWorkflowType, workflowType],
  );

  const contextValue = useMemo(
    () => ({
      workflowType,
      setWorkflowType: handleWorkflowTypeChange,
    }),
    [workflowType, handleWorkflowTypeChange],
  );

  return <WorkflowTypeContext.Provider value={contextValue}>{children}</WorkflowTypeContext.Provider>;
};

export const useWorkflowType = (): WorkflowTypeContextType => {
  return useContext(WorkflowTypeContext);
};
