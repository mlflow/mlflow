import type { FormInstance as AntDFormInstance, FormItemProps as AntDFormItemProps, FormProps as AntDFormProps } from 'antd';
import { Form as AntDForm } from 'antd';
import type { FC } from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type LegacyFormInstance = AntDFormInstance;
export interface LegacyFormItemProps extends Pick<AntDFormItemProps, 'children' | 'dependencies' | 'getValueFromEvent' | 'getValueProps' | 'help' | 'hidden' | 'htmlFor' | 'initialValue' | 'label' | 'messageVariables' | 'name' | 'normalize' | 'noStyle' | 'preserve' | 'required' | 'rules' | 'shouldUpdate' | 'tooltip' | 'trigger' | 'validateStatus' | 'validateTrigger' | 'valuePropName'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDFormItemProps> {
}
export interface LegacyFormProps extends Pick<AntDFormProps, 'children' | 'className' | 'component' | 'fields' | 'form' | 'initialValues' | 'labelCol' | 'layout' | 'name' | 'preserve' | 'requiredMark' | 'validateMessages' | 'validateTrigger' | 'wrapperCol' | 'onFieldsChange' | 'onFinish' | 'onFinishFailed' | 'onValuesChange'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDFormProps> {
}
interface FormInterface extends FC<LegacyFormProps> {
    Item: typeof FormItem;
    List: typeof AntDForm.List;
    useForm: typeof AntDForm.useForm;
}
/**
 * @deprecated Use `Form` from `@databricks/design-system/development` instead.
 */
export declare const LegacyFormDubois: import("react").ForwardRefExoticComponent<LegacyFormProps & import("react").RefAttributes<LegacyFormInstance>>;
declare const FormItem: FC<LegacyFormItemProps>;
export declare const LegacyForm: FormInterface;
export declare const __INTERNAL_DO_NOT_USE__FormItem: FC<LegacyFormItemProps>;
export {};
//# sourceMappingURL=LegacyForm.d.ts.map