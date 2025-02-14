import React, { useContext } from 'react';

interface DesignSystemEventSuppressInteractionContextType {
  suppressAnalyticsStartInteraction: boolean;
}

const DefaultDesignSystemEventSuppressInteractionContextValue: DesignSystemEventSuppressInteractionContextType = {
  suppressAnalyticsStartInteraction: false,
};

export const DesignSystemEventSuppressInteractionTrueContextValue: DesignSystemEventSuppressInteractionContextType = {
  suppressAnalyticsStartInteraction: true,
};

export const DesignSystemEventSuppressInteractionProviderContext =
  React.createContext<DesignSystemEventSuppressInteractionContextType>(
    DefaultDesignSystemEventSuppressInteractionContextValue,
  );

/**
 * This gets the event suppress interaction provider, is used to suppress the start interaction for analytics events.
 *
 * @returns DesignSystemEventSuppressInteractionContextType
 */
export const useDesignSystemEventSuppressInteractionContext = () => {
  return useContext(DesignSystemEventSuppressInteractionProviderContext);
};
