import type { CSSObject, Interpolation, SerializedStyles, Theme as EmotionTheme } from '@emotion/react';
import { css } from '@emotion/react';
import type { CollapseProps as AntDCollapseProps, CollapsePanelProps as AntDCollapsePanelProps } from 'antd';
import { Collapse as AntDCollapse } from 'antd';
import type { CollapsibleType } from 'antd/lib/collapse/CollapsePanel';
import { useCallback, useMemo } from 'react';

import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronDownIcon } from '../Icon';
import type { AnalyticsEventValueChangeNoPiiFlagProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineStylesIfEnabled } from '../utils/debug';

export interface AccordionProps
  extends HTMLDataAttributes,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /** How many sections can be displayed at once */
  displayMode?: 'single' | 'multiple';
  /** Key of the active panel */
  activeKey?: Array<string | number> | string | number;
  /** Specify whether the entire header (`undefined` (default)) or children (`"header"`) are the collapsible trigger. `"disabled"` disables the collapsible behavior of the accordion headers. */
  collapsible?: CollapsibleType;
  /** Key of the initial active panel */
  defaultActiveKey?: Array<string | number> | string | number;
  /** Destroy Inactive Panel */
  destroyInactivePanel?: boolean;
  /** Callback function executed when active panel is changed */
  onChange?: (key: string | string[]) => void;
  /** Escape hatch to allow passing props directly to the underlying Ant `TabPane` component. */
  dangerouslySetAntdProps?: Partial<AntDCollapseProps>;
  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface AccordionPanelProps extends HTMLDataAttributes {
  children: React.ReactNode;
  /** Unique key identifying the panel from among its siblings */
  key: string | number;
  /** Title of the panel */
  header: React.ReactNode;
  /** Specify whether the entire header (`undefined` (default)) or children (`"header"`) are the collapsible trigger. `"disabled"` disables the collapsible behavior of the accordion headers. */
  collapsible?: CollapsibleType;
  /** Forced render of content on panel, instead of lazy rending after clicking on header */
  forceRender?: boolean;
  /** Escape hatch to allow passing props directly to the underlying Ant `TabPane` component. */
  dangerouslySetAntdProps?: Partial<AntDCollapsePanelProps>;
  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

interface AccordionInterface extends React.FC<AccordionProps> {
  Panel: React.FC<AccordionPanelProps>;
}

function getAccordionEmotionStyles(clsPrefix: string, theme: Theme): SerializedStyles {
  const classItem = `.${clsPrefix}-item`;
  const classItemActive = `${classItem}-active`;
  const classHeader = `.${clsPrefix}-header`;
  const classContent = `.${clsPrefix}-content`;
  const classContentBox = `.${clsPrefix}-content-box`;
  const classArrow = `.${clsPrefix}-arrow`;

  const styles: CSSObject = {
    border: '0 none',
    background: 'none',

    [classItem]: {
      border: '0 none',

      [`&:hover`]: {
        [classHeader]: {
          color: theme.colors.actionPrimaryBackgroundHover,
        },

        [classArrow]: {
          color: theme.colors.actionPrimaryBackgroundHover,
        },
      },

      [`&:active`]: {
        [classHeader]: {
          color: theme.colors.actionPrimaryBackgroundPress,
        },

        [classArrow]: {
          color: theme.colors.actionPrimaryBackgroundPress,
        },
      },
    },

    [classHeader]: {
      color: theme.colors.textPrimary,
      fontWeight: 600,

      '&:focus-visible': {
        outlineColor: `${theme.colors.actionDefaultBorderFocus} !important`,
        outlineStyle: 'auto !important',
      },
    },

    [`& > ${classItem} > ${classHeader} > ${classArrow}`]: {
      fontSize: theme.general.iconFontSize,
      right: 12,
    },

    [classArrow]: {
      color: theme.colors.textSecondary,
    },

    [`& > ${classItemActive} > ${classHeader} > ${classArrow}`]: {
      transform: 'translateY(-50%) rotate(180deg)',
    },

    [classContent]: {
      border: '0 none',
      backgroundColor: theme.colors.backgroundPrimary,
    },

    [classContentBox]: {
      padding: '8px 16px 16px',
    },

    [`& > ${classItem} > ${classHeader}`]: {
      padding: '6px 44px 6px 0',
      lineHeight: theme.typography.lineHeightBase,
    },

    ...getAnimationCss(theme.options.enableAnimation),
  };

  return css(styles);
}

export const AccordionPanel: React.FC<AccordionPanelProps> = ({
  dangerouslySetAntdProps,
  dangerouslyAppendEmotionCSS,
  children,
  ...props
}: AccordionPanelProps) => {
  return (
    <DesignSystemAntDConfigProvider>
      <AntDCollapse.Panel {...props} {...dangerouslySetAntdProps} css={dangerouslyAppendEmotionCSS}>
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </AntDCollapse.Panel>
    </DesignSystemAntDConfigProvider>
  );
};

export const Accordion = /* #__PURE__ */ (() => {
  const Accordion: AccordionInterface = ({
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    displayMode = 'multiple',
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    componentId,
    valueHasNoPii,
    onChange,
    ...props
  }) => {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    // While this component is called `Accordion` for correctness, in AntD it is called `Collapse`.
    const clsPrefix = getPrefixedClassName('collapse');
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.Accordion,
      componentId,
      analyticsEvents: memoizedAnalyticsEvents,
      valueHasNoPii,
    });

    const onChangeWrapper = useCallback(
      (newValue: string | string[]) => {
        if (Array.isArray(newValue)) {
          eventContext.onValueChange(JSON.stringify(newValue));
        } else {
          eventContext.onValueChange(newValue);
        }

        onChange?.(newValue);
      },
      [eventContext, onChange],
    );

    return (
      <DesignSystemAntDConfigProvider>
        <AntDCollapse
          // eslint-disable-next-line @databricks/no-unstable-nested-components -- go/no-nested-components
          expandIcon={() => <ChevronDownIcon {...eventContext.dataComponentProps} />}
          expandIconPosition="right"
          accordion={displayMode === 'single'}
          {...props}
          {...dangerouslySetAntdProps}
          css={[
            getAccordionEmotionStyles(clsPrefix, theme),
            dangerouslyAppendEmotionCSS,
            addDebugOutlineStylesIfEnabled(theme),
          ]}
          onChange={onChangeWrapper}
        />
      </DesignSystemAntDConfigProvider>
    );
  };
  Accordion.Panel = AccordionPanel;

  return Accordion;
})();
