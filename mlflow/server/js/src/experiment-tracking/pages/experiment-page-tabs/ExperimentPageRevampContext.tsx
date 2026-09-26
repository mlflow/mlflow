import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ExperimentKind } from '../../constants';
import { useLocation } from '../../../common/utils/RoutingUtils';

interface PublishedTitle {
  routeKey: string;
  title?: string;
}

interface ExperimentPageRevampContextValue {
  enabled: boolean;
  inferredExperimentKind?: ExperimentKind;
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
  inferredExperimentKind?: ExperimentKind;
}

export const ExperimentPageRevampProvider = ({
  children,
  enabled,
  inferredExperimentKind,
}: ExperimentPageRevampProviderProps) => {
  const [publishedTitle, setPublishedTitle] = useState<PublishedTitle>();
  const value = useMemo(
    () => ({ enabled, inferredExperimentKind, publishedTitle, setPublishedTitle }),
    [enabled, inferredExperimentKind, publishedTitle],
  );

  return <ExperimentPageRevampContext.Provider value={value}>{children}</ExperimentPageRevampContext.Provider>;
};

export const useExperimentPageRevampContext = () => useContext(ExperimentPageRevampContext);

export const useExperimentBreadcrumbTitle = () => {
  const { pathname } = useLocation();
  const { enabled, publishedTitle } = useExperimentPageRevampContext();
  return enabled && publishedTitle?.routeKey === pathname ? publishedTitle.title : undefined;
};

export const useSetExperimentBreadcrumbTitle = (title?: string) => {
  const { pathname } = useLocation();
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
      setPublishedTitle({ routeKey: pathname, title });
    } else {
      clearTitle();
    }
    return clearTitle;
  }, [pathname, clearTitle, enabled, setPublishedTitle, title]);
};
