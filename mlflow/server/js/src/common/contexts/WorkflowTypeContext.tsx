import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { useNavigate, useParams } from '../utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import { useGetExperimentQuery } from '../../experiment-tracking/hooks/useExperimentQuery';
import {
  getExperimentKindFromTagsList,
  getWorkflowTypeForExperimentKind,
} from '../../experiment-tracking/utils/ExperimentKindUtils';

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

  // Sync the workflow type to the experiment the user actually opened, so that
  // following a shared link into an experiment shows the correct sidebar/toggle
  // instead of whatever was last persisted (defaulting to GENAI). We key the
  // sync on the experiment id: it runs once per experiment, letting a manual
  // toggle override stick while the user stays on that experiment, and only
  // re-reconciling when they navigate to a different one. This uses the plain
  // ``setWorkflowType`` (not the navigating ``handleWorkflowTypeChange``) so the
  // reconciliation never bounces the user off their deep link.
  const { data: experiment } = useGetExperimentQuery({ experimentId });
  const lastSyncedExperimentIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (!experimentId) {
      // Reset when leaving experiment scope so re-entering the same experiment
      // reconciles again.
      lastSyncedExperimentIdRef.current = undefined;
      return;
    }
    if (lastSyncedExperimentIdRef.current === experimentId) {
      return;
    }
    // Wait until the query has loaded the experiment that matches the current
    // route. Apollo keeps returning the previously-observed experiment until the
    // new one resolves, so guarding on the id prevents syncing to a stale kind
    // (which would also pin the ref and never self-correct).
    if (!experiment || experiment.experimentId !== experimentId) {
      return;
    }
    const experimentKind = getExperimentKindFromTagsList(experiment.tags);
    const targetWorkflowType = getWorkflowTypeForExperimentKind(experimentKind);
    lastSyncedExperimentIdRef.current = experimentId;
    if (targetWorkflowType && targetWorkflowType !== workflowType) {
      setWorkflowType(targetWorkflowType);
    }
  }, [experimentId, experiment, workflowType, setWorkflowType]);

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
