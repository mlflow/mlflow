import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useLocation } from '../../../common/utils/RoutingUtils';

interface PublishedTitle {
  routeKey: string;
  title?: string;
  status?: string;
}

interface ExperimentPageRevampContextValue {
  enabled: boolean;
  publishedTitle?: PublishedTitle;
  setPublishedTitle: React.Dispatch<React.SetStateAction<PublishedTitle | undefined>>;
}

const ExperimentPageRevampContext = createContext<ExperimentPageRevampContextValue>({
  enabled: false,
  setPublishedTitle: () => undefined,
});

export interface ExperimentPageRevampProviderProps {
  children: React.ReactNode;
  enabled: boolean;
}

export const ExperimentPageRevampProvider = ({ children, enabled }: ExperimentPageRevampProviderProps) => {
  const [publishedTitle, setPublishedTitle] = useState<PublishedTitle>();
  const value = useMemo(() => ({ enabled, publishedTitle, setPublishedTitle }), [enabled, publishedTitle]);

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
