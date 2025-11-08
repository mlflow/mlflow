import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';

import {
  GearIcon,
  ListBorderIcon,
  ListIcon,
  ModelsIcon,
  PlusMinusSquareIcon,
  UserIcon,
  TextBoxIcon,
  SparkleIcon,
} from '@databricks/design-system';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { FormattedMessage } from 'react-intl';
import type { ExperimentViewRunsCompareMode } from '@mlflow/mlflow/src/experiment-tracking/types';

export type TabConfig = {
  label: React.ReactNode;
  icon: React.ReactNode;
  getRoute: (experimentId: string) => string;
};

export type TabConfigMap = Partial<Record<ExperimentViewRunsCompareMode | ExperimentPageTabName, TabConfig>>;

const RunsTabConfig = {
  label: (
    <FormattedMessage defaultMessage="Runs" description="Label for the runs tab in the MLflow experiment navbar" />
  ),
  icon: <ListIcon />,
  getRoute: (experimentId: string) => Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Runs),
};

const TracesTabConfig = {
  label: (
    <FormattedMessage defaultMessage="Traces" description="Label for the traces tab in the MLflow experiment navbar" />
  ),
  icon: <ListBorderIcon />,
  getRoute: (experimentId: string) => Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces),
};

const EvaluationsTabConfig = {
  label: (
    <FormattedMessage
      defaultMessage="Evaluations"
      description="Label for the evaluations tab in the MLflow experiment navbar"
    />
  ),
  icon: <PlusMinusSquareIcon />,
  getRoute: (experimentId: string) =>
    Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns),
};

const ModelsTabConfig = {
  label: (
    <FormattedMessage
      defaultMessage="Versions"
      description="Label for the logged models tab in the MLflow experiment navbar"
    />
  ),
  icon: <ModelsIcon />,
  getRoute: (experimentId: string) => Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models),
};

export type GenAIExperimentTabConfigMapProps = {
  includeRunsTab?: boolean;
};

export const getGenAIExperimentTabConfigMap = ({
  includeRunsTab = false,
}: GenAIExperimentTabConfigMapProps = {}): TabConfigMap => ({
  ...(includeRunsTab && { [ExperimentPageTabName.Runs]: RunsTabConfig }),
  [ExperimentPageTabName.Traces]: TracesTabConfig,
  [ExperimentPageTabName.EvaluationRuns]: EvaluationsTabConfig,
  [ExperimentPageTabName.Models]: ModelsTabConfig,
});

export const getGenAIExperimentWithPromptsTabConfigMap = ({
  includeRunsTab = false,
}: GenAIExperimentTabConfigMapProps = {}): TabConfigMap => ({
  ...(includeRunsTab && { [ExperimentPageTabName.Runs]: RunsTabConfig }),
  [ExperimentPageTabName.Traces]: TracesTabConfig,
  [ExperimentPageTabName.Models]: ModelsTabConfig,
});

export const GenAIExperimentWithPromptsTabConfigMap = getGenAIExperimentTabConfigMap();

export const CustomExperimentTabConfigMap: TabConfigMap = {
  [ExperimentPageTabName.Runs]: RunsTabConfig,
  [ExperimentPageTabName.Models]: {
    ...ModelsTabConfig,
    label: (
      <FormattedMessage
        defaultMessage="Models"
        description="Label for the logged models tab in the MLflow experiment navbar"
      />
    ),
  },
  [ExperimentPageTabName.Traces]: TracesTabConfig,
};

export const DefaultTabConfigMap: TabConfigMap = {
  ...CustomExperimentTabConfigMap,
};
