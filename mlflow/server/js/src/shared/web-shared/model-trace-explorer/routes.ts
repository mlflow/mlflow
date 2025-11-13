/**
 * This file contains a subset of mlflow/web/js/src/experiment-tracking/routes.tsx to be used in the model-trace-explorer.
 */
import { createMLflowRoutePath, generatePath } from './RoutingUtils';

// Route path definitions (used in defining route elements)
const createExperimentPageRoutePath = () => createMLflowRoutePath('/experiments/:experimentId');

export const getExperimentPageRoute = (experimentId: string) => {
  return generatePath(createExperimentPageRoutePath(), { experimentId });
};

export const getExperimentPageTracesTabRoute = (experimentId: string) => {
  return `${getExperimentPageRoute(experimentId)}/traces`;
};
