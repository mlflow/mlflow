import type { CSSObject } from '@emotion/react';
import type { TabsProps as AntDTabsProps, TabPaneProps as AntDTabPaneProps } from 'antd';
import { Tabs as AntDTabs } from 'antd';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon, PlusIcon } from '../Icon';
import type { DangerousGeneralProps, HTMLDataAttributes } from '../types';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export const getLegacyTabEmotionStyles = (clsPrefix: string, theme: Theme): CSSObject => {
  const classTab = `.${clsPrefix}-tabs-tab`;
  const classButton = `.${clsPrefix}-tabs-tab-btn`;
  const classActive = `.${clsPrefix}-tabs-tab-active`;
  const classDisabled = `.${clsPrefix}-tabs-tab-disabled`;
  const classUnderline = `.${clsPrefix}-tabs-ink-bar`;
  const classClosable = `.${clsPrefix}-tabs-tab-with-remove`;
  const classNav = `.${clsPrefix}-tabs-nav`;
  const classCloseButton = `.${clsPrefix}-tabs-tab-remove`;
  const classAddButton = `.${clsPrefix}-tabs-nav-add`;

  const styles: CSSObject = {
    '&&': {
      overflow: 'unset',
    },

    [classTab]: {
      borderBottom: 'none',
      backgroundColor: 'transparent',
      border: 'none',
      paddingLeft: 0,
      paddingRight: 0,
      paddingTop: 6,
      paddingBottom: 6,
      marginRight: 24,
    },

    [classButton]: {
      color: theme.colors.textSecondary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      textShadow: 'none',
      fontSize: theme.typography.fontSizeMd,
      lineHeight: theme.typography.lineHeightBase,

      '&:hover': {
        color: theme.colors.actionDefaultTextHover,
      },
      '&:active': {
        color: theme.colors.actionDefaultTextPress,
      },

      outlineWidth: 2,
      outlineStyle: 'none',
      outlineColor: theme.colors.actionDefaultBorderFocus,
      outlineOffset: 2,

      '&:focus-visible': {
        outlineStyle: 'auto',
      },
    },

    [classActive]: {
      [classButton]: {
        color: theme.colors.textPrimary,
      },
      // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
      // jumping when switching tabs.
      boxShadow: `inset 0 -3px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,
    },

    [classDisabled]: {
      [classButton]: {
        color: theme.colors.actionDisabledText,
        '&:hover': {
          color: theme.colors.actionDisabledText,
        },
        '&:active': {
          color: theme.colors.actionDisabledText,
        },
      },
    },

    [classUnderline]: {
      display: 'none',
    },

    [classClosable]: {
      borderTop: 'none',
      borderLeft: 'none',
      borderRight: 'none',
      background: 'none',
      paddingTop: 0,
      paddingBottom: 0,
      height: theme.general.heightSm,
    },

    [classNav]: {
      height: theme.general.heightSm,
      '&::before': {
        borderColor: theme.colors.borderDecorative,
      },
    },

    [classCloseButton]: {
      height: 24,
      width: 24,
      padding: 6,
      borderRadius: theme.legacyBorders.borderRadiusMd,
      marginTop: 0,
      marginRight: 0,
      marginBottom: 0,
      marginLeft: 4,

      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover,
      },

      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress,
      },

      '&:focus-visible': {
        outlineWidth: 2,
        outlineStyle: 'solid',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },
    },

    [classAddButton]: {
      backgroundColor: 'transparent',
      color: theme.colors.textValidationInfo,
      border: 'none',
      borderRadius: theme.legacyBorders.borderRadiusMd,
      margin: 4,
      height: 24,
      width: 24,
      padding: 0,
      minWidth: 'auto',

      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover,
      },

      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress,
      },

      '&:focus-visible': {
        outlineWidth: 2,
        outlineStyle: 'solid',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },

      '& > .anticon': {
        fontSize: 16,
      },
    },

    ...getAnimationCss(theme.options.enableAnimation),
  };

  const importantStyles = importantify(styles);

  return importantStyles;
};

/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export interface LegacyTabsProps extends HTMLDataAttributes, DangerousGeneralProps {
  /**
   * Set this to `true` to allow users to add and remove tab panes.
   */
  editable?: boolean;

  /**
   * Determines the active tab pane.
   * Use this for controlled mode to handle the active state yourself.
   */
  activeKey?: AntDTabsProps['activeKey'];

  /**
   * Determines the tab pane that is initially active before the user interacts.
   * Use this for uncontrolled mode to let the component handle the active state itself.
   */
  defaultActiveKey?: AntDTabsProps['defaultActiveKey'];

  /**
   * Called when the user clicks on a tab. Use this in combination with `activeKey` for a controlled component.
   */
  onChange?: AntDTabsProps['onChange'];

  /**
   * Called when the user clicks the add or delete buttons. Use in combination with `editable=true`.
   */
  onEdit?: AntDTabsProps['onEdit'];

  /**
   * One or more instances of <TabPane /> to render inside this tab container.
   */
  children?: AntDTabsProps['children'];

  /**
   *  Whether destroy inactive TabPane when change tab
   */
  destroyInactiveTabPane?: boolean;

  /**
   * Escape hatch to allow passing props directly to the underlying Ant `Tabs` component.
   */
  dangerouslySetAntdProps?: Partial<AntDTabsProps>;

  // Forces CSSObject type to support importantify
  dangerouslyAppendEmotionCSS?: CSSObject;
}

/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export interface LegacyTabPaneProps {
  /**
   * Text to display in the table title.
   */
  tab: AntDTabPaneProps['tab'];

  /**
   * Whether or not this tab is disabled.
   */
  disabled?: AntDTabPaneProps['disabled'];

  /**
   * Content to render inside the tab body.
   */
  children?: AntDTabPaneProps['children'];

  /**
   * Escape hatch to allow passing props directly to the underlying Ant `TabPane` component.
   */
  dangerouslySetAntdProps?: Partial<AntDTabPaneProps>;
}

/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export interface LegacyTabsInterface extends React.FC<LegacyTabsProps> {
  TabPane: typeof LegacyTabPane;
}

/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export const LegacyTabPane: React.FC<LegacyTabPaneProps> = ({ children, ...props }: LegacyTabPaneProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDTabs.TabPane
        closeIcon={<CloseIcon css={{ fontSize: theme.general.iconSize }} />}
        // Note: this component must accept the entire `props` object and spread it here, because Ant's Tabs components
        // injects extra props here (at the time of writing, `prefixCls`, `tabKey` and `id`).
        // However, we use a restricted TS interface to still discourage consumers of the library from passing in these props.
        {...props}
        {...props.dangerouslySetAntdProps}
      >
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </AntDTabs.TabPane>
    </DesignSystemAntDConfigProvider>
  );
};

/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export const LegacyTabs = /* #__PURE__ */ (() => {
  const LegacyTabs: LegacyTabsInterface = ({
    editable = false,
    activeKey,
    defaultActiveKey,
    onChange,
    onEdit,
    children,
    destroyInactiveTabPane = false,
    dangerouslySetAntdProps = {},
    dangerouslyAppendEmotionCSS = {},
    ...props
  }) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();

    return (
      <DesignSystemAntDConfigProvider>
        <AntDTabs
          {...addDebugOutlineIfEnabled()}
          activeKey={activeKey}
          defaultActiveKey={defaultActiveKey}
          onChange={onChange}
          onEdit={onEdit}
          destroyInactiveTabPane={destroyInactiveTabPane}
          type={editable ? 'editable-card' : 'card'}
          addIcon={<PlusIcon css={{ fontSize: theme.general.iconSize }} />}
          css={[getLegacyTabEmotionStyles(classNamePrefix, theme), importantify(dangerouslyAppendEmotionCSS)]}
          {...dangerouslySetAntdProps}
          {...props}
        >
          {children}
        </AntDTabs>
      </DesignSystemAntDConfigProvider>
    );
  };

  LegacyTabs.TabPane = LegacyTabPane;

  return LegacyTabs;
})();
