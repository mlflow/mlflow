import type { Input as AntDInput, InputProps as AntDInputProps } from 'antd';
import type { GroupProps as AntDGroupProps, PasswordProps as AntDPasswordProps, TextAreaProps as AntDTextAreaProps } from 'antd/lib/input';
import type { TextAreaRef } from 'antd/lib/input/TextArea';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { DangerousGeneralProps, DangerouslySetAntdProps, FormElementValidationState, HTMLDataAttributes } from '../types';
export type InputRef = AntDInput;
export type { TextAreaRef };
export interface InputProps extends Omit<AntDInputProps, 'prefixCls' | 'size' | 'addonAfter' | 'bordered'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDInputProps>, DangerousGeneralProps {
    onClear?: () => void;
    /**
     * componentId is used to identify the component in analytics events. It distinguished
     * this component from all other components of this type. For new instances of this
     * component it's good to have this be a short, human-readable string that describes
     * the component. For example, the syntax for the identifier could be something similar
     * to "webapp.notebook.share".
     * This will be used in querying component events in analytics.
     * go/ui-observability
     */
    componentId?: string;
    analyticsEvents?: ReadonlyArray<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange>;
}
export interface PasswordProps extends Omit<AntDPasswordProps, 'inputPrefixCls' | 'action' | 'visibilityToggle' | 'iconRender'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDPasswordProps>, DangerousGeneralProps {
}
export interface TextAreaProps extends Omit<AntDTextAreaProps, 'bordered' | 'showCount' | 'size'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDTextAreaProps>, DangerousGeneralProps {
}
export interface InputGroupProps extends Omit<AntDGroupProps, 'size'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDGroupProps>, DangerousGeneralProps {
}
//# sourceMappingURL=common.d.ts.map