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
import { PageWrapper, useDesignSystemTheme } from '@databricks/design-system';

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
  const { theme } = useDesignSystemTheme();

  useEffect(() => {
    const pageTitle = formatMessage({
      defaultMessage: 'Experiment Runs - Databricks',
      description: 'Title on a page used to manage MLflow experiments runs',
    });
    Utils.updatePageTitle(pageTitle);
  }, [formatMessage]);

  return (
    <PageWrapper css={{ height: '100%', paddingTop: theme.spacing.md }}>
      <GetExperimentsContextProvider actions={getExperimentActions}>
        <ExperimentView />
      </GetExperimentsContextProvider>
    </PageWrapper>
  );
};

export default ExperimentPage;
