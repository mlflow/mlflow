import { useSelectedIssueId } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId';

/**
 * The issue the page has selected, or `undefined` where there is no router to read it from.
 *
 * `useSelectedIssueId` reads the query string through `useSearchParams`, which throws outside a
 * `<Router>`. The trace explorer is mounted without one in the OSS notebook renderer and in several
 * test harnesses, so a surface that read the selection directly would crash there rather than
 * simply having no selection.
 *
 * The `try` does not make the hook call conditional -- it runs on every render, so the hook order is
 * the same whether or not a router is present. It only decides what happens to the error.
 */
export const useSafeSelectedIssueId = (): string | undefined => {
  try {
    const [selectedIssueId] = useSelectedIssueId();
    return selectedIssueId;
  } catch {
    return undefined;
  }
};
