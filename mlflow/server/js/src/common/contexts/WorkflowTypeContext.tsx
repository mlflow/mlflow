import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { useNavigate, useParams } from '../utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import { useGetExperimentQuery } from '../../experiment-tracking/hooks/useExperimentQuery';
import {
  getExperimentKindFromTags,
  getWorkflowTypeForExperimentKind,
} from '../../experiment-tracking/utils/ExperimentKindUtils';

export enum WorkflowType {
  GENAI = 'genai',
  MACHINE_LEARNING = 'machine_learning',
}

export const WORKFLOW_TYPE_STORAGE_KEY = 'mlflow.workflowType';
export const WORKFLOW_TYPE_STORAGE_VERSION = 1;
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

  // Records the experiment the user manually toggled within, so auto-sync does
  // not clobber a deliberate choice (even if the experiment's kind tag lands or
  // changes later while they stay on that experiment).
  const manualOverrideExperimentIdRef = useRef<string | undefined>(undefined);
  const handleWorkflowTypeChange = useCallback(
    (newWorkflowType: WorkflowType) => {
      if (newWorkflowType === workflowType) {
        return;
      }

      if (experimentId) {
        manualOverrideExperimentIdRef.current = experimentId;
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
  // instead of whatever was last persisted (defaulting to GENAI). This uses the
  // plain ``setWorkflowType`` (not the navigating ``handleWorkflowTypeChange``)
  // so the reconciliation never bounces the user off their deep link.
  const { data: experiment } = useGetExperimentQuery({ experimentId });
  // Tracks the experiment we have already reconciled to a definite kind. We only
  // lock this once a definite mapping is applied, so an experiment whose kind tag
  // is populated later (e.g. by kind inference) still reconciles rather than
  // being permanently skipped.
  const lastSyncedExperimentIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (!experimentId) {
      // Reset when leaving experiment scope so re-entering an experiment
      // reconciles again.
      lastSyncedExperimentIdRef.current = undefined;
      manualOverrideExperimentIdRef.current = undefined;
      return;
    }
    // A manual toggle within this experiment wins; do not auto-reconcile over it.
    if (manualOverrideExperimentIdRef.current === experimentId) {
      return;
    }
    if (lastSyncedExperimentIdRef.current === experimentId) {
      return;
    }
    // Wait until the query has loaded the experiment that matches the current
    // route. Apollo keeps returning the previously-observed experiment until the
    // new one resolves, so guarding on the id prevents syncing to a stale kind.
    if (!experiment || experiment.experimentId !== experimentId) {
      return;
    }
    const experimentKind = getExperimentKindFromTags(experiment.tags);
    const targetWorkflowType = getWorkflowTypeForExperimentKind(experimentKind);
    // Only lock (and sync) once we have a definite mapping. An absent/ambiguous
    // kind leaves the current selection untouched and stays eligible to
    // reconcile if the tag is populated later while on the same experiment.
    if (!targetWorkflowType) {
      return;
    }
    lastSyncedExperimentIdRef.current = experimentId;
    if (targetWorkflowType !== workflowType) {
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
