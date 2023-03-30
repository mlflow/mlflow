import { useEffect } from 'react';
import { useIntl } from 'react-intl';
// prettier-ignore
import {
  getExperimentApi,
  setCompareExperiments,
  setExperimentTagApi,
} from '../../actions';
import Utils from '../../../common/utils/Utils';
import { GetExperimentsContextProvider } from './contexts/GetExperimentsContext';
import { ExperimentView } from './ExperimentView';

/**
 * Concrete actions for GetExperiments context
 */
const getExperimentActions = {
  setExperimentTagApi,
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
