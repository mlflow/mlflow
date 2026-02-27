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
import { LegacySkeleton, PageWrapper, useDesignSystemTheme } from '@databricks/design-system';
import { useNavigateToExperimentPageTab } from './hooks/useNavigateToExperimentPageTab';
import { useExperimentIds } from './hooks/useExperimentIds';

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
const ExperimentPage = () => {
  const { formatMessage } = useIntl();
  const { theme } = useDesignSystemTheme();
  const experimentIds = useExperimentIds();

  useEffect(() => {
    const pageTitle = formatMessage({
      defaultMessage: 'Experiment Runs - Databricks',
      description: 'Title on a page used to manage MLflow experiments runs',
    });
    Utils.updatePageTitle(pageTitle);
  }, [formatMessage]);

  const isComparingExperiments = experimentIds.length > 1;

  // Check if view mode determines rendering using another route. If true, wait for the redirection and return null.
  const { isLoading: isAutoNavigatingToTab, isEnabled: isAutoNavigateEnabled } = useNavigateToExperimentPageTab({
    enabled: !isComparingExperiments,
    experimentId: experimentIds[0],
  });

  if (isAutoNavigatingToTab) {
    return <LegacySkeleton />;
  }
  if (isAutoNavigateEnabled) {
    return null;
  }
  return (
    <PageWrapper css={{ height: '100%', paddingTop: theme.spacing.md }}>
      <GetExperimentsContextProvider actions={getExperimentActions}>
        <ExperimentView />
      </GetExperimentsContextProvider>
    </PageWrapper>
  );
};

export default ExperimentPage;
