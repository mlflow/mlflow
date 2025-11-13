import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Collapse as AntDCollapse } from 'antd';
import { useCallback, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronDownIcon } from '../Icon';
import { useNotifyOnFirstView } from '../utils';
import { addDebugOutlineStylesIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
function getAccordionEmotionStyles({ clsPrefix, theme, alignContentToEdge, isLeftAligned, }) {
    const classItem = `.${clsPrefix}-item`;
    const classItemActive = `${classItem}-active`;
    const classHeader = `.${clsPrefix}-header`;
    const classContent = `.${clsPrefix}-content`;
    const classContentBox = `.${clsPrefix}-content-box`;
    const classArrow = `.${clsPrefix}-arrow`;
    const styles = {
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
            right: alignContentToEdge || isLeftAligned ? 0 : 12,
            ...(isLeftAligned && {
                verticalAlign: 'middle',
                marginTop: -2,
            }),
        },
        [classArrow]: {
            color: theme.colors.textSecondary,
        },
        [`& > ${classItemActive} > ${classHeader} > ${classArrow}`]: {
            transform: isLeftAligned ? 'rotate(180deg)' : 'translateY(-50%) rotate(180deg)',
        },
        [classContent]: {
            border: '0 none',
            backgroundColor: theme.colors.backgroundPrimary,
        },
        [classContentBox]: {
            padding: alignContentToEdge ? '8px 0px 16px' : '8px 16px 16px',
        },
        [`& > ${classItem} > ${classHeader}`]: {
            padding: '6px 44px 6px 0',
            lineHeight: theme.typography.lineHeightBase,
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    return css(styles);
}
export const AccordionPanel = ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, children, ...props }) => {
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDCollapse.Panel, { ...props, ...dangerouslySetAntdProps, css: dangerouslyAppendEmotionCSS, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
};
export const Accordion = /* #__PURE__ */ (() => {
    const Accordion = ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, displayMode = 'multiple', analyticsEvents, componentId, valueHasNoPii, onChange, alignContentToEdge = false, chevronAlignment = 'right', ...props }) => {
        const emitOnView = safex('databricks.fe.observability.defaultComponentView.accordion', false);
        const { theme, getPrefixedClassName } = useDesignSystemTheme();
        // While this component is called `Accordion` for correctness, in AntD it is called `Collapse`.
        const clsPrefix = getPrefixedClassName('collapse');
        const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
            (emitOnView
                ? [
                    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                    DesignSystemEventProviderAnalyticsEventTypes.OnView,
                ]
                : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
        const eventContext = useDesignSystemEventComponentCallbacks({
            componentType: DesignSystemEventProviderComponentTypes.Accordion,
            componentId,
            analyticsEvents: memoizedAnalyticsEvents,
            valueHasNoPii,
        });
        const { elementRef: accordionRef } = useNotifyOnFirstView({
            onView: eventContext.onView,
        });
        const onChangeWrapper = useCallback((newValue) => {
            if (Array.isArray(newValue)) {
                eventContext.onValueChange(JSON.stringify(newValue));
            }
            else {
                eventContext.onValueChange(newValue);
            }
            onChange?.(newValue);
        }, [eventContext, onChange]);
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDCollapse
            // eslint-disable-next-line @databricks/no-unstable-nested-components -- go/no-nested-components
            , { 
                // eslint-disable-next-line @databricks/no-unstable-nested-components -- go/no-nested-components
                expandIcon: () => _jsx(ChevronDownIcon, { ...eventContext.dataComponentProps, ref: accordionRef }), expandIconPosition: chevronAlignment, accordion: displayMode === 'single', ...props, ...dangerouslySetAntdProps, css: [
                    getAccordionEmotionStyles({
                        clsPrefix,
                        theme,
                        alignContentToEdge,
                        isLeftAligned: chevronAlignment === 'left',
                    }),
                    dangerouslyAppendEmotionCSS,
                    addDebugOutlineStylesIfEnabled(theme),
                ], onChange: onChangeWrapper }) }));
    };
    Accordion.Panel = AccordionPanel;
    return Accordion;
})();
//# sourceMappingURL=Accordion.js.map