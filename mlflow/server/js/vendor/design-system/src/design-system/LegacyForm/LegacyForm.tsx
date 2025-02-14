import { css } from '@emotion/react';
import type {
  FormInstance as AntDFormInstance,
  FormItemProps as AntDFormItemProps,
  FormProps as AntDFormProps,
} from 'antd';
import { Form as AntDForm } from 'antd';
import { forwardRef } from 'react';
import type { FC } from 'react';

import type { ComponentTheme } from '../../theme';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export type LegacyFormInstance = AntDFormInstance;

const getFormItemEmotionStyles = ({ theme, clsPrefix }: { theme: ComponentTheme; clsPrefix: string }) => {
  const clsFormItemLabel = `.${clsPrefix}-form-item-label`;
  const clsFormItemInputControl = `.${clsPrefix}-form-item-control-input`;
  const clsFormItemExplain = `.${clsPrefix}-form-item-explain`;
  const clsHasError = `.${clsPrefix}-form-item-has-error`;

  return css({
    [clsFormItemLabel]: {
      fontWeight: theme.typography.typographyBoldFontWeight,
      lineHeight: theme.typography.lineHeightBase,

      '.anticon': {
        fontSize: theme.general.iconFontSize,
      },
    },
    [clsFormItemExplain]: {
      fontSize: theme.typography.fontSizeSm,
      margin: 0,

      [`&${clsFormItemExplain}-success`]: {
        color: theme.colors.textValidationSuccess,
      },
      [`&${clsFormItemExplain}-warning`]: {
        color: theme.colors.textValidationDanger,
      },
      [`&${clsFormItemExplain}-error`]: {
        color: theme.colors.textValidationDanger,
      },
      [`&${clsFormItemExplain}-validating`]: {
        color: theme.colors.textSecondary,
      },
    },

    [clsFormItemInputControl]: {
      minHeight: theme.general.heightSm,
    },
    [`${clsFormItemInputControl} input[disabled]`]: {
      border: 'none',
    },
    [`&${clsHasError} input:focus`]: importantify({
      boxShadow: 'none',
    }),
    ...getAnimationCss(theme.options.enableAnimation),
  });
};

export interface LegacyFormItemProps
  extends Pick<
      AntDFormItemProps,
      | 'children'
      | 'dependencies'
      | 'getValueFromEvent'
      | 'getValueProps'
      | 'help'
      | 'hidden'
      | 'htmlFor'
      | 'initialValue'
      | 'label'
      | 'messageVariables'
      | 'name'
      | 'normalize'
      | 'noStyle'
      | 'preserve'
      | 'required'
      | 'rules'
      | 'shouldUpdate'
      | 'tooltip'
      | 'trigger'
      | 'validateStatus'
      | 'validateTrigger'
      | 'valuePropName'
    >,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDFormItemProps> {}

export interface LegacyFormProps
  extends Pick<
      AntDFormProps,
      | 'children'
      | 'className'
      | 'component'
      | 'fields'
      | 'form'
      | 'initialValues'
      | 'labelCol'
      | 'layout'
      | 'name'
      | 'preserve'
      | 'requiredMark'
      | 'validateMessages'
      | 'validateTrigger'
      | 'wrapperCol'
      | 'onFieldsChange'
      | 'onFinish'
      | 'onFinishFailed'
      | 'onValuesChange'
    >,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDFormProps> {}

interface FormInterface extends FC<LegacyFormProps> {
  Item: typeof FormItem;
  List: typeof AntDForm.List;
  useForm: typeof AntDForm.useForm;
}

/**
 * @deprecated Use `Form` from `@databricks/design-system/development` instead.
 */
export const LegacyFormDubois = forwardRef<LegacyFormInstance, LegacyFormProps>(function Form(
  { dangerouslySetAntdProps, children, ...props },
  ref,
) {
  const mergedProps: LegacyFormProps = {
    ...props,
    layout: props.layout || 'vertical',
    requiredMark: props.requiredMark || false,
  };

  return (
    <DesignSystemAntDConfigProvider>
      <AntDForm {...addDebugOutlineIfEnabled()} {...mergedProps} colon={false} ref={ref} {...dangerouslySetAntdProps}>
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </AntDForm>
    </DesignSystemAntDConfigProvider>
  );
});

const FormItem: FC<LegacyFormItemProps> = ({ dangerouslySetAntdProps, children, ...props }) => {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  return (
    <DesignSystemAntDConfigProvider>
      <AntDForm.Item
        {...addDebugOutlineIfEnabled()}
        {...props}
        css={getFormItemEmotionStyles({
          theme,
          clsPrefix: classNamePrefix,
        })}
        {...dangerouslySetAntdProps}
      >
        {children}
      </AntDForm.Item>
    </DesignSystemAntDConfigProvider>
  );
};

const FormNamespace = /* #__PURE__ */ Object.assign(LegacyFormDubois, {
  Item: FormItem,
  List: AntDForm.List,
  useForm: AntDForm.useForm,
});
export const LegacyForm: FormInterface = FormNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
export const __INTERNAL_DO_NOT_USE__FormItem = FormItem;
