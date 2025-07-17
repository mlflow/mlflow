import React, { useContext } from 'react';
const DefaultDesignSystemEventSuppressInteractionContextValue = {
    suppressAnalyticsStartInteraction: false,
};
export const DesignSystemEventSuppressInteractionTrueContextValue = {
    suppressAnalyticsStartInteraction: true,
};
export const DesignSystemEventSuppressInteractionProviderContext = React.createContext(DefaultDesignSystemEventSuppressInteractionContextValue);
/**
 * This gets the event suppress interaction provider, is used to suppress the start interaction for analytics events.
 *
 * @returns DesignSystemEventSuppressInteractionContextType
 */
export const useDesignSystemEventSuppressInteractionContext = () => {
    return useContext(DesignSystemEventSuppressInteractionProviderContext);
};
//# sourceMappingURL=DesignSystemEventSuppressInteractionProvider.js.map