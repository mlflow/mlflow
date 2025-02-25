import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import type { DropdownMenuProps } from '@radix-ui/react-dropdown-menu';
import { useCallback, useState } from 'react';
import type { ReactElement } from 'react';

import { DropdownButton } from './Dropdown/DropdownButton';
import type { DropdownButtonProps } from './Dropdown/DropdownButton';
import type { Theme } from '../../theme';
import { Button } from '../Button';
import {
  getDefaultStyles,
  getDisabledPrimarySplitButtonStyles,
  getDisabledSplitButtonStyles,
  getPrimaryStyles,
} from '../Button/styles';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronDownIcon } from '../Icon';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';

// Taken from rc-menu
// TODO: Check if we can maybe depend on rc-menu?
export interface SplitButtonMenuInfo {
  key: React.Key;
  keyPath: React.Key[];
  domEvent: React.SyntheticEvent<HTMLElement>;
}
export type SplitButtonProps = Omit<DropdownButtonProps, 'overlay' | 'type' | 'size' | 'trigger'> &
  HTMLDataAttributes &
  DangerouslySetAntdProps<Partial<DropdownButtonProps>> & {
    /**
     * @deprecated Please migrate to the DuBois DropdownMenu component and use the `menu` prop.
     */
    deprecatedMenu?: DropdownButtonProps['overlay'];
    /**
     * The visual style of the button, either default or primary
     */
    type?: 'default' | 'primary';
    loading?: boolean;
    loadingButtonStyles?: React.CSSProperties;
    /**
     * Props to be passed down to DropdownMenu.Root
     */
    dropdownMenuRootProps?: DropdownMenuProps;
  };
const BUTTON_HORIZONTAL_PADDING = 12;
function getSplitButtonEmotionStyles(classNamePrefix: string, theme: Theme, useNewShadows: boolean): SerializedStyles {
  const classDefault = `.${classNamePrefix}-btn`;
  const classPrimary = `.${classNamePrefix}-btn-primary`;
  const classDropdownTrigger = `.${classNamePrefix}-dropdown-trigger`;
  const classSmall = `.${classNamePrefix}-btn-group-sm`;
  const styles: CSSObject = {
    [classDefault]: {
      ...getDefaultStyles(theme),

      boxShadow: useNewShadows ? theme.shadows.xs : 'none',
      height: theme.general.heightSm,
      padding: `4px ${BUTTON_HORIZONTAL_PADDING}px`,

      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },
      '.anticon, &:focus-visible .anticon': {
        color: theme.colors.textSecondary,
      },
      '&:hover .anticon': {
        color: theme.colors.actionDefaultIconHover,
      },
      '&:active .anticon': {
        color: theme.colors.actionDefaultIconPress,
      },
    },

    [classPrimary]: {
      ...getPrimaryStyles(theme),
      ...(useNewShadows && {
        boxShadow: theme.shadows.xs,
      }),

      [`&:first-of-type`]: {
        borderRight: `1px solid ${theme.colors.actionPrimaryTextDefault}`,
        marginRight: 1,
      },
      [classDropdownTrigger]: {
        borderLeft: `1px solid ${theme.colors.actionPrimaryTextDefault}`,
      },
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '1px',
        outlineOffset: '-3px',
        outlineColor: theme.colors.white,
      },
      '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
        color: theme.colors.actionPrimaryIcon,
      },
    },
    [classDropdownTrigger]: {
      // Needs to be 1px less than our standard 8px to allow for the off-by-one border handling in this component.
      padding: 3,
      borderLeftColor: 'transparent',
      width: theme.general.heightSm,
    },
    [`&${classSmall}`]: {
      [classDropdownTrigger]: {
        padding: 5,
      },
    },

    '&&': {
      [`[disabled], ${classPrimary}[disabled]`]: {
        ...getDisabledSplitButtonStyles(theme, useNewShadows),

        ...(useNewShadows && {
          boxShadow: 'none',
        }),

        [`&:first-of-type`]: {
          borderRight: `1px solid ${theme.colors.actionPrimaryIcon}`,
          marginRight: 1,
        },
        [classDropdownTrigger]: {
          borderLeft: `1px solid ${theme.colors.actionPrimaryIcon}`,
        },
        '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
          color: theme.colors.actionDisabledText,
        },
      },

      [`${classPrimary}[disabled]`]: {
        ...getDisabledPrimarySplitButtonStyles(theme, useNewShadows),
        '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
          color: theme.colors.actionPrimaryTextDefault,
        },
      },
    },
    [`${classDefault}:not(:first-of-type)`]: {
      width: theme.general.heightSm,
      padding: '3px !important',
    },
    ...getAnimationCss(theme.options.enableAnimation),
  };
  const importantStyles = importantify(styles);
  return css(importantStyles);
}
export const SplitButton: React.FC<SplitButtonProps> = (props: SplitButtonProps): ReactElement => {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const {
    children,
    icon,
    deprecatedMenu,
    type,
    loading,
    loadingButtonStyles,
    placement,
    dangerouslySetAntdProps,
    ...dropdownButtonProps
  } = props;
  // Size of button when loading only icon is shown
  const LOADING_BUTTON_SIZE =
    theme.general.iconFontSize + 2 * BUTTON_HORIZONTAL_PADDING + 2 * theme.general.borderWidth;
  const [width, setWidth] = useState(LOADING_BUTTON_SIZE);
  // Set the width to the button's width in regular state to later use when in loading state
  // We do this to have just a loading icon in loading state at the normal width to avoid flicker and width changes in page
  const ref = useCallback(
    (node: HTMLDivElement) => {
      if (node && !loading) {
        setWidth(node.getBoundingClientRect().width);
      }
    },
    [loading],
  );
  return (
    <DesignSystemAntDConfigProvider>
      <div ref={ref} css={{ display: 'inline-flex', position: 'relative', verticalAlign: 'middle' }}>
        {loading ? (
          <Button
            componentId="codegen_design-system_src_design-system_splitbutton_splitbutton.tsx_163"
            type={type === 'default' ? undefined : type}
            style={{
              width: width,
              fontSize: theme.general.iconFontSize,
              ...loadingButtonStyles,
            }}
            loading
            htmlType={props.htmlType}
            title={props.title}
            className={props.className}
          >
            {children}
          </Button>
        ) : (
          <DropdownButton
            {...dropdownButtonProps}
            overlay={deprecatedMenu}
            trigger={['click']}
            css={getSplitButtonEmotionStyles(classNamePrefix, theme, useNewShadows)}
            icon={<ChevronDownIcon css={{ fontSize: theme.general.iconFontSize }} aria-hidden="true" />}
            placement={placement || 'bottomRight'}
            type={type === 'default' ? undefined : type}
            leftButtonIcon={icon}
            {...dangerouslySetAntdProps}
          >
            {children}
          </DropdownButton>
        )}
      </div>
    </DesignSystemAntDConfigProvider>
  );
};
