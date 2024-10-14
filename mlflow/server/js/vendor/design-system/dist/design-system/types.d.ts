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
export interface AnalyticsEventValueChangeNoPii {
    /**
     * Opt-in flag to indicate that the value change event should emit the value, and has no concerns for PII in the value.
     *
     * The values should likely be enum strings or a constrained set of string values that are not PII.
     *
     * By default is false, but will still emit the event with the value being empty as long as componentId is provided.
     */
    valueHasNoPii?: boolean;
}
export interface AnalyticsEventProps<T extends DesignSystemEventProviderAnalyticsEventTypes> {
    /**
     * componentId is used to identify the component in analytics events. It distinguished
     * this component from all other components of this type. For new instances of this
     * component it's good to have this be a short, human-readable string that describes
     * the component. For example, the syntax for the identifier could be something similar
     * to "webapp.notebook.share".
     * This will be used in querying component events in analytics.
     * go/product-analytics
     */
    componentId: string;
    analyticsEvents?: ReadonlyArray<T>;
}
export interface AnalyticsEventValueChangeNoPiiFlagProps<T extends DesignSystemEventProviderAnalyticsEventTypes> extends AnalyticsEventProps<T>, AnalyticsEventValueChangeNoPii {
}
export interface AnalyticsEventOptionalProps<T extends DesignSystemEventProviderAnalyticsEventTypes> {
    /**
     * componentId is used to identify the component in analytics events. It distinguished
     * this component from all other components of this type. For new instances of this
     * component it's good to have this be a short, human-readable string that describes
     * the component. For example, the syntax for the identifier could be something similar
     * to "webapp.notebook.share".
     * This will be used in querying component events in analytics.
     * go/product-analytics
     */
    componentId?: string;
    analyticsEvents?: ReadonlyArray<T>;
}
export interface AnalyticsEventValueChangeNoPiiFlagOptionalProps<T extends DesignSystemEventProviderAnalyticsEventTypes> extends AnalyticsEventOptionalProps<T>, AnalyticsEventValueChangeNoPii {
}
export interface AnalyticsEventPropsWithStartInteraction<T extends DesignSystemEventProviderAnalyticsEventTypes> extends AnalyticsEventProps<T> {
    /**
     * Flag to indicate if an RIT interaction should be started when an event is triggered.
     *
     * For the Table & Modal components, any children components will default to false, otherwise will default to true.
     * This is due to bad patterns these components have had that should be resolved with this lint introduced in FEINF-3364
     */
    shouldStartInteraction?: boolean;
}
export interface AnalyticsEventOptionalPropsWithStartInteraction<T extends DesignSystemEventProviderAnalyticsEventTypes> extends AnalyticsEventOptionalProps<T> {
    /**
     * Flag to indicate if an RIT interaction should be started when an event is triggered.
     *
     * For the Table & Modal components, any children components will default to false, otherwise will default to true.
     * This is due to bad patterns these components have had that should be resolved with this lint introduced in FEINF-3364
     */
    shouldStartInteraction?: boolean;
}
//# sourceMappingURL=types.d.ts.map