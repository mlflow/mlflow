// prettier-ignore
import {
  deleteExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
  setExperimentTagApi
} from '../../actions';

import { ExperimentView } from './ExperimentView';
import { GetExperimentsContextProvider } from './contexts/GetExperimentsContext';
import Utils from '../../../common/utils/Utils';
import { useEffect } from 'react';
import { useIntl } from 'react-intl';

/**
 * Concrete actions for GetExperiments context
 */
const getExperimentActions = {
  setExperimentTagApi,
  deleteExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

/**
 * Main entry point for the experiment page. This component
 * provides underlying structure with context containing
 * concrete versions of store actions.
 */
export const ExperimentPage = () => {
  const { formatMessage } = useIntl();

  useEffect(() => {
    const pageTitle = formatMessage({
      defaultMessage: 'Experiment Runs - Databricks',
      description: 'Title on a page used to manage MLflow experiments runs',
    });
    Utils.updatePageTitle(pageTitle);
  });

  return (
    <GetExperimentsContextProvider actions={getExperimentActions}>
      <ExperimentView />
    </GetExperimentsContextProvider>
  );
};

export default ExperimentPage;
