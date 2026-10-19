import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useLocation } from '../../../common/utils/RoutingUtils';

interface PublishedTitle {
  routeKey: string;
  title?: string;
  status?: string;
}

interface PublishedPairwiseReviewAction {
  routeKey: string;
  action: () => void;
}

interface ExperimentPageRevampContextValue {
  enabled: boolean;
  publishedTitle?: PublishedTitle;
  setPublishedTitle: React.Dispatch<React.SetStateAction<PublishedTitle | undefined>>;
  publishedPairwiseReviewAction?: PublishedPairwiseReviewAction;
  setPublishedPairwiseReviewAction: React.Dispatch<React.SetStateAction<PublishedPairwiseReviewAction | undefined>>;
}

const ExperimentPageRevampContext = createContext<ExperimentPageRevampContextValue>({
  enabled: false,
  setPublishedTitle: () => undefined,
  setPublishedPairwiseReviewAction: () => undefined,
});

export interface ExperimentPageRevampProviderProps {
  children: React.ReactNode;
  enabled: boolean;
}

export const ExperimentPageRevampProvider = ({ children, enabled }: ExperimentPageRevampProviderProps) => {
  const [publishedTitle, setPublishedTitle] = useState<PublishedTitle>();
  const [publishedPairwiseReviewAction, setPublishedPairwiseReviewAction] = useState<PublishedPairwiseReviewAction>();
  const value = useMemo(
    () => ({
      enabled,
      publishedTitle,
      setPublishedTitle,
      publishedPairwiseReviewAction,
      setPublishedPairwiseReviewAction,
    }),
    [enabled, publishedTitle, publishedPairwiseReviewAction],
  );

  return <ExperimentPageRevampContext.Provider value={value}>{children}</ExperimentPageRevampContext.Provider>;
};

export const useExperimentPageRevampContext = () => useContext(ExperimentPageRevampContext);

export const useExperimentBreadcrumbTitle = () => {
  const pathname = useLocation().pathname;
  const { enabled, publishedTitle } = useExperimentPageRevampContext();
  return enabled && publishedTitle?.routeKey === pathname ? publishedTitle.title : undefined;
};

export const useExperimentBreadcrumbStatus = () => {
  const pathname = useLocation().pathname;
  const { enabled, publishedTitle } = useExperimentPageRevampContext();
  return enabled && publishedTitle?.routeKey === pathname ? publishedTitle.status : undefined;
};

/** The current route's published "start pairwise review" action, shown as Review in the breadcrumb row. */
export const useExperimentBreadcrumbPairwiseReviewAction = () => {
  const pathname = useLocation().pathname;
  const { enabled, publishedPairwiseReviewAction } = useExperimentPageRevampContext();
  return enabled && publishedPairwiseReviewAction?.routeKey === pathname
    ? publishedPairwiseReviewAction.action
    : undefined;
};

/**
 * Publishes a display name for the current route's breadcrumb leaf, for example a dataset name in place
 * of its ID, with an optional status shown next to the breadcrumb. Does nothing unless the experiment page
 * shell is enabled.
 */
export const useSetExperimentBreadcrumbTitle = (title?: string, status?: string) => {
  const pathname = useLocation().pathname;
  const { enabled, setPublishedTitle } = useExperimentPageRevampContext();

  const clearTitle = useCallback(
    () => setPublishedTitle((current) => (current?.routeKey === pathname ? undefined : current)),
    [pathname, setPublishedTitle],
  );

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    if (title) {
      setPublishedTitle({ routeKey: pathname, title, status });
    } else {
      clearTitle();
    }
    return clearTitle;
  }, [pathname, clearTitle, enabled, setPublishedTitle, title, status]);
};

/** Publishes the current route's "start pairwise review" action; leaving the route or passing undefined clears it. */
export const useSetExperimentBreadcrumbPairwiseReviewAction = (action?: () => void) => {
  const pathname = useLocation().pathname;
  const { enabled, setPublishedPairwiseReviewAction } = useExperimentPageRevampContext();

  const clearAction = useCallback(
    () => setPublishedPairwiseReviewAction((current) => (current?.routeKey === pathname ? undefined : current)),
    [pathname, setPublishedPairwiseReviewAction],
  );

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    if (action) {
      setPublishedPairwiseReviewAction({ routeKey: pathname, action });
    } else {
      clearAction();
    }
    return clearAction;
  }, [action, clearAction, enabled, pathname, setPublishedPairwiseReviewAction]);
};
