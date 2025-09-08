import { merge } from 'lodash';
import type { ReactNode } from 'react';
import React, { createContext, useContext, useMemo, useCallback } from 'react';

// A global config that is used as a context for monitoring components.
export interface MonitoringConfig {
  dateNow: Date;
  lastRefreshTime: number;
  refresh: () => void;
}

// Define a default configuration
const getDefaultConfig = (): MonitoringConfig => {
  return {
    dateNow: new Date(),
    lastRefreshTime: Date.now(),
    refresh: () => {},
  };
};

// Create the context with a default value
const MonitoringConfigContext = createContext<MonitoringConfig>(getDefaultConfig());

interface MonitoringConfigProviderProps {
  config?: Partial<MonitoringConfig>;
  children: ReactNode;
}

export const MonitoringConfigProvider: React.FC<React.PropsWithChildren<MonitoringConfigProviderProps>> = ({
  config,
  children,
}) => {
  const defaultConfig = getDefaultConfig();
  // Remove undefined values from the config object

  const mergedConfig = merge({}, defaultConfig, config);

  const [lastRefreshTime, setLastRefreshTime] = React.useState(mergedConfig.lastRefreshTime);

  // Derive dateNow from lastRefreshTime
  const dateNow = useMemo(() => new Date(lastRefreshTime), [lastRefreshTime]);

  // Single refresh method
  const refresh = useCallback(() => {
    setLastRefreshTime(Date.now());
  }, []);

  return (
    <MonitoringConfigContext.Provider
      value={{
        ...mergedConfig,
        dateNow,
        lastRefreshTime,
        refresh,
      }}
    >
      {children}
    </MonitoringConfigContext.Provider>
  );
};

export const useMonitoringConfig = (): MonitoringConfig => {
  const context = useContext(MonitoringConfigContext);

  if (!context) {
    return getDefaultConfig(); // Fallback to defaults if no provider is found
  }

  return context;
};
