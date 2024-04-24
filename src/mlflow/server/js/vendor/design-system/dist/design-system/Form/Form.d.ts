import type { FormInstance as AntDFormInstance, FormItemProps as AntDFormItemProps, FormProps as AntDFormProps } from 'antd';
import { Form as AntDForm } from 'antd';
import type { FC } from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type FormInstance = AntDFormInstance;
export interface FormItemProps extends Pick<AntDFormItemProps, 'children' | 'dependencies' | 'getValueFromEvent' | 'getValueProps' | 'help' | 'hidden' | 'htmlFor' | 'initialValue' | 'label' | 'messageVariables' | 'name' | 'normalize' | 'noStyle' | 'preserve' | 'required' | 'rules' | 'shouldUpdate' | 'tooltip' | 'trigger' | 'validateStatus' | 'validateTrigger' | 'valuePropName'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDFormItemProps> {
}
export interface FormProps extends Pick<AntDFormProps, 'children' | 'className' | 'component' | 'fields' | 'form' | 'initialValues' | 'labelCol' | 'layout' | 'name' | 'preserve' | 'requiredMark' | 'validateMessages' | 'validateTrigger' | 'wrapperCol' | 'onFieldsChange' | 'onFinish' | 'onFinishFailed' | 'onValuesChange'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDFormProps> {
}
interface FormInterface extends FC<FormProps> {
    Item: typeof FormItem;
    List: typeof AntDForm.List;
    useForm: typeof AntDForm.useForm;
}
export declare const FormDubois: import("react").ForwardRefExoticComponent<FormProps & import("react").RefAttributes<FormInstance>>;
declare const FormItem: FC<FormItemProps>;
export declare const Form: FormInterface;
export declare const __INTERNAL_DO_NOT_USE__FormItem: FC<FormItemProps>;
export {};
//# sourceMappingURL=Form.d.ts.map