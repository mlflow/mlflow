import { useEffect } from 'react';
import { useMatches } from '../utils/RoutingUtils';

/**
 * Hook that updates the document title based on the current route's handle metadata.
 * Routes can specify a title in their handle property:
 *
 * @example
 * {
 *   path: '/experiments',
 *   element: <ExperimentPage />,
 *   pageId: 'mlflow.experiments.list',
 *   handle: { title: 'Experiments' }
 * }
 *
 * The title will be displayed as "Title - MLflow" in the browser tab.
 */
export const useDocumentTitle = () => {
  const matches = useMatches();

  useEffect(() => {
    // Find the last route match that has a title in its handle
    const lastMatchWithTitle = [...matches].reverse().find((match) => {
      const handle = match.handle as { title?: string } | undefined;
      return handle?.title;
    });

    if (lastMatchWithTitle) {
      const handle = lastMatchWithTitle.handle as { title: string };
      document.title = `${handle.title} - MLflow`;
    } else {
      // Fallback to default title if no route has a title
      document.title = 'MLflow';
    }
  }, [matches]);
};
