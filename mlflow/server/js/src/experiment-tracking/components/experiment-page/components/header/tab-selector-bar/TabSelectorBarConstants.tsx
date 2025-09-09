import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';

import {
  GearIcon,
  ListBorderIcon,
  ListIcon,
  ModelsIcon,
  PlusMinusSquareIcon,
  UserIcon,
  TextBoxIcon,
} from '@databricks/design-system';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { FormattedMessage } from 'react-intl';
import { ExperimentViewRunsCompareMode } from '@mlflow/mlflow/src/experiment-tracking/types';

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

export const GenAIExperimentTabConfigMap: TabConfigMap = {
  [ExperimentPageTabName.Traces]: TracesTabConfig,
  [ExperimentPageTabName.Models]: ModelsTabConfig,
};

export const GenAIExperimentWithPromptsTabConfigMap = GenAIExperimentTabConfigMap;

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
