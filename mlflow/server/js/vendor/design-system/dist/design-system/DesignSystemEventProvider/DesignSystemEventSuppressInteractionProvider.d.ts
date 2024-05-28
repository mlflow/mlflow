import React from 'react';
interface DesignSystemEventSuppressInteractionContextType {
    suppressAnalyticsStartInteraction: boolean;
}
export declare const DesignSystemEventSuppressInteractionTrueContextValue: DesignSystemEventSuppressInteractionContextType;
export declare const DesignSystemEventSuppressInteractionProviderContext: React.Context<DesignSystemEventSuppressInteractionContextType>;
/**
 * This gets the event suppress interaction provider, is used to suppress the start interaction for analytics events.
 *
 * @returns DesignSystemEventSuppressInteractionContextType
 */
export declare const useDesignSystemEventSuppressInteractionContext: () => DesignSystemEventSuppressInteractionContextType;
export {};
//# sourceMappingURL=DesignSystemEventSuppressInteractionProvider.d.ts.map