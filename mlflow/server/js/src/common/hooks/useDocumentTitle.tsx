import { useEffect } from 'react';
import { useMatches, type Params } from '../utils/RoutingUtils';

/**
 * Type for the getPageTitle function that can be added to route handle.
 */
export type GetPageTitleFn = (params: Params<string>) => string;

/**
 * Route handle type for document title configuration.
 */
export interface DocumentTitleHandle {
  getPageTitle?: GetPageTitleFn;
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
 * The title will be displayed as "Title - MLflow" in the browser tab.
 */
export const useDocumentTitle = () => {
  const matches = useMatches();

  useEffect(() => {
    // Find the last route match that has a getPageTitle function in its handle
    const lastMatchWithTitle = [...matches].reverse().find((match) => {
      const handle = match.handle as DocumentTitleHandle | undefined;
      return typeof handle?.getPageTitle === 'function';
    });

    if (lastMatchWithTitle) {
      const handle = lastMatchWithTitle.handle as DocumentTitleHandle;
      const title = handle.getPageTitle!(lastMatchWithTitle.params);
      document.title = `${title} - MLflow`;
    } else {
      // Fallback to default title if no route has a title
      document.title = 'MLflow';
    }
  }, [matches]);
};
