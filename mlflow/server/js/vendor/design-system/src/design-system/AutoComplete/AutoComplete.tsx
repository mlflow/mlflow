import { css } from '@emotion/react';
import type { AutoCompleteProps as AntDAutoCompleteProps } from 'antd';
import { AutoComplete as AntDAutoComplete } from 'antd';
import type { OptionType } from 'antd/es/select';

import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { getDarkModePortalStyles, useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export interface AutoCompleteProps extends AntDAutoCompleteProps {}

export interface AutoCompleteProps
  extends AntDAutoCompleteProps,
    DangerouslySetAntdProps<AntDAutoCompleteProps>,
    HTMLDataAttributes {}

interface AutoCompleteInterface extends React.FC<AutoCompleteProps> {
  Option: OptionType;
}

/**
 * @deprecated Use `TypeaheadCombobox` instead.
 */
export const AutoComplete = /* #__PURE__ */ (() => {
  const AutoComplete: AutoCompleteInterface = ({ dangerouslySetAntdProps, ...props }) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();

    return (
      <DesignSystemAntDConfigProvider>
        <AntDAutoComplete
          {...addDebugOutlineIfEnabled()}
          dropdownStyle={{ boxShadow: theme.general.shadowLow, ...getDarkModePortalStyles(theme, useNewShadows) }}
          {...props}
          {...dangerouslySetAntdProps}
          css={css(getAnimationCss(theme.options.enableAnimation))}
        />
      </DesignSystemAntDConfigProvider>
    );
  };

  AutoComplete.Option = AntDAutoComplete.Option;

  return AutoComplete;
})();
