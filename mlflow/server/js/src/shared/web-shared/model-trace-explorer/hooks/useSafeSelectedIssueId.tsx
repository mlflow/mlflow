import { useSelectedIssueId } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId';

import { useInRouterContext } from '../RoutingUtils';

/**
 * The issue the page has selected, or `undefined` where there is no router to read it from.
 *
 * `useSelectedIssueId` reads the query string through `useSearchParams`, which throws outside a
 * `<Router>`. The trace explorer is mounted without one in the OSS notebook renderer and in several
 * test harnesses, so a surface that read the selection directly would crash there rather than
 * simply having no selection.
 *
 * `useInRouterContext` narrows the `catch` to exactly that case: with a router present -- which is
 * every real page -- anything `useSelectedIssueId` throws is a genuine failure and is rethrown
 * rather than silently becoming "no selection". Without one, `useSearchParams` throws its invariant
 * before the hook does any other work, so there is nothing else the error could be.
 *
 * The `try` does not make the hook calls conditional -- both run on every render, so the hook order
 * is the same whether or not a router is present. It only decides what happens to the error.
 */
export const useSafeSelectedIssueId = (): string | undefined => {
  const inRouterContext = useInRouterContext();

  try {
    const [selectedIssueId] = useSelectedIssueId();
    return selectedIssueId;
  } catch (error) {
    if (inRouterContext) {
      throw error;
    }
    return undefined;
  }
};
