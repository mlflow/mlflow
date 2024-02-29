import type { Interpolation } from '@emotion/react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from './DesignSystemEventProvider';
import type { ColorVars } from './constants';
import type { Theme } from '../theme';
export interface DangerouslySetAntdProps<P> {
    /** For components that wrap `antd` components, emergency access for properties we do not support. Ask in #dubois before using. */
    dangerouslySetAntdProps?: P;
}
export interface DangerousGeneralProps {
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<Theme>;
}
/** Mapping for color variables can be found under constants.tsx */
export type TypographyColor = keyof typeof ColorVars;
/** Generic type for supporting data- attributes */
export interface HTMLDataAttributes {
    [key: `data-${string}`]: string;
}
export type ValidationState = 'success' | 'warning' | 'error';
export interface FormElementValidationState {
    validationState?: ValidationState;
}
export interface AnalyticsEventProps<T extends DesignSystemEventProviderAnalyticsEventTypes> {
    /**
     * componentId is used to identify the component in analytics events. It distinguished
     * this component from all other components of this type. For new instances of this
     * component it's good to have this be a short, human-readable string that describes
     * the component. For example, the syntax for the identifier could be something similar
     * to "webapp.notebook.share".
     * This will be used in querying component events in analytics.
     * go/ui-observability
     */
    componentId: string;
    analyticsEvents?: ReadonlyArray<T>;
}
//# sourceMappingURL=types.d.ts.map