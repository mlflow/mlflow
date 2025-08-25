import { merge } from 'lodash';
import type { ReactNode } from 'react';
import React, { createContext, useContext } from 'react';

// A global config that is used as a context for monitoring components.
export interface MonitoringConfig {
  dateNow: Date;
  setDateNow: (date: Date) => void;
}

// Define a default configuration
const getDefaultConfig = (): MonitoringConfig => {
  return {
    dateNow: new Date(),
    setDateNow: (date: Date) => {},
  };
};

// Create the context with a default value
const MonitoringConfigContext = createContext<MonitoringConfig>(getDefaultConfig());

interface MonitoringConfigProviderProps {
  config?: Partial<MonitoringConfig>;
  children: ReactNode;
}

export const MonitoringConfigProvider: React.FC<MonitoringConfigProviderProps> = ({ config, children }) => {
  const defaultConfig = getDefaultConfig();
  // Remove undefined values from the config object

  const mergedConfig: MonitoringConfig = merge({}, defaultConfig, config);

  const [dateNow, setDateNow] = React.useState(mergedConfig.dateNow);

  return (
    <MonitoringConfigContext.Provider
      value={{
        ...mergedConfig,
        dateNow,
        setDateNow,
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
