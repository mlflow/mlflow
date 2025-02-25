import { css } from '@emotion/react';
import { Input as AntDInput } from 'antd';

import type { InputGroupProps } from './common';
import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const getInputGroupStyling = (clsPrefix: string, theme: Theme, useNewShadows: boolean) => {
  const inputClass = `.${clsPrefix}-input`;
  const buttonClass = `.${clsPrefix}-btn`;
  return css({
    display: 'inline-flex !important',
    width: 'auto',
    [`& > ${inputClass}`]: {
      flexGrow: 1,
      '&:disabled': {
        border: 'none',
        background: theme.colors.actionDisabledBackground,
        '&:hover': {
          borderRight: `1px solid ${theme.colors.actionDisabledBorder} !important`,
        },
      },
      '&[data-validation]': {
        marginRight: 0,
      },
    },

    ...(useNewShadows && {
      [`& > ${buttonClass}`]: {
        boxShadow: 'none !important',
      },
    }),

    [`& > ${buttonClass} > span`]: {
      verticalAlign: 'middle',
    },
    [`& > ${buttonClass}:disabled, & > ${buttonClass}:disabled:hover`]: {
      borderLeft: `1px solid ${theme.colors.actionDisabledBorder} !important`,
      backgroundColor: `${theme.colors.actionDisabledBackground} !important`,
      color: `${theme.colors.actionDisabledText} !important`,
    },
  });
};
export const Group = ({
  dangerouslySetAntdProps,
  dangerouslyAppendEmotionCSS,
  compact = true,
  ...props
}: InputGroupProps) => {
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDInput.Group
        {...addDebugOutlineIfEnabled()}
        css={[getInputGroupStyling(classNamePrefix, theme, useNewShadows), dangerouslyAppendEmotionCSS]}
        compact={compact}
        {...props}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
};
