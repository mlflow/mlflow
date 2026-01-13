import { useEffect } from 'react';
import { useLocation, useMatches, type Params } from '../utils/RoutingUtils';

/**
 * Type for the getPageTitle function that can be added to route handle.
 */
export type GetPageTitleFn = (params: Params<string>) => string;

/**
 * Route handle type for document title configuration.
 */
export interface DocumentTitleHandle {
  getPageTitle: GetPageTitleFn;
}

/**
 * Hook that updates the document title based on the current route's handle.getPageTitle function.
 * Routes can specify a getPageTitle function in their handle property:
 *
 * @example
 * {
 *   path: '/experiments/:experimentId',
 *   element: <ExperimentPage />,
 *   pageId: 'mlflow.experiment.details',
 *   handle: { getPageTitle: (params) => `Experiment ${params['experimentId']}` }
 * }
 *
 * The title will be displayed as "Experiment 123 - MLflow" in the browser tab.
 */
export const useDocumentTitle = () => {
  const matches = useMatches();

  useEffect(() => {
    if (matches.length === 0) {
      return;
    }

    const lastMatch = matches[matches.length - 1];
    const handle = lastMatch.handle as DocumentTitleHandle | undefined;
    const title = handle?.getPageTitle(lastMatch.params);

    if (title) {
      document.title = `${title} - MLflow`;
    } else {
      document.title = 'MLflow';
    }
  }, [matches]);
};
