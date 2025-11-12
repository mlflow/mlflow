import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { Pagination as AntdPagination } from 'antd';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronLeftIcon, ChevronRightIcon } from '../Icon';
import { LegacySelect } from '../LegacySelect';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
export function getPaginationEmotionStyles(clsPrefix, theme, useNewShadows, useNewBorderColors) {
    const classRoot = `.${clsPrefix}-pagination`;
    const classItem = `.${clsPrefix}-pagination-item`;
    const classLink = `.${clsPrefix}-pagination-item-link`;
    const classActive = `.${clsPrefix}-pagination-item-active`;
    const classEllipsis = `.${clsPrefix}-pagination-item-ellipsis`;
    const classNext = `.${clsPrefix}-pagination-next`;
    const classPrev = `.${clsPrefix}-pagination-prev`;
    const classJumpNext = `.${clsPrefix}-pagination-jump-next`;
    const classJumpPrev = `.${clsPrefix}-pagination-jump-prev`;
    const classQuickJumper = `.${clsPrefix}-pagination-options-quick-jumper`;
    const classSizeChanger = `.${clsPrefix}-pagination-options-size-changer`;
    const classOptions = `.${clsPrefix}-pagination-options`;
    const classDisabled = `.${clsPrefix}-pagination-disabled`;
    const classSelector = `.${clsPrefix}-select-selector`;
    const classDropdown = `.${clsPrefix}-select-dropdown`;
    const styles = {
        'span[role=img]': {
            color: theme.colors.textSecondary,
            '> *': {
                color: 'inherit',
            },
        },
        [classItem]: {
            backgroundColor: 'none',
            border: 'none',
            color: theme.colors.textSecondary,
            '&:focus-visible': {
                outline: 'auto',
            },
            '> a': {
                color: theme.colors.textSecondary,
                textDecoration: 'none',
                '&:hover': {
                    color: theme.colors.actionDefaultTextHover,
                },
                '&:active': {
                    color: theme.colors.actionDefaultTextPress,
                },
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
        },
        [classActive]: {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            color: theme.colors.actionDefaultTextPress,
            border: 'none',
            '> a': {
                color: theme.colors.actionDefaultTextPress,
            },
            '&:focus-visible': {
                outline: 'auto',
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress,
            },
        },
        [classLink]: {
            border: 'none',
            color: theme.colors.textSecondary,
            '&[disabled]': {
                display: 'none',
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
            '&:focus-visible': {
                outline: 'auto',
            },
        },
        [classEllipsis]: {
            color: 'inherit',
        },
        [`${classNext}, ${classPrev}, ${classJumpNext}, ${classJumpPrev}`]: {
            color: theme.colors.textSecondary,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
            '&:focus-visible': {
                outline: 'auto',
            },
            [`&${classDisabled}`]: {
                pointerEvents: 'none',
            },
        },
        [`&${classRoot}.mini, ${classRoot}.mini`]: {
            [`${classItem}, ${classNext}, ${classPrev}, ${classJumpNext}, ${classJumpPrev}`]: {
                height: '32px',
                minWidth: '32px',
                width: 'auto',
                lineHeight: '32px',
            },
            [classSizeChanger]: {
                marginLeft: 4,
            },
            [`input,  ${classOptions}`]: {
                height: '32px',
            },
            [`${classSelector}`]: {
                ...(useNewShadows && {
                    boxShadow: theme.shadows.xs,
                }),
                ...(useNewBorderColors && {
                    borderColor: theme.colors.actionDefaultBorderDefault,
                }),
            },
        },
        ...(useNewBorderColors && {
            [`${classQuickJumper} > input`]: {
                borderColor: theme.colors.actionDefaultBorderDefault,
            },
            [`${classDropdown}`]: {
                borderColor: theme.colors.actionDefaultBorderDefault,
            },
        }),
    };
    const importantStyles = importantify(styles);
    return css(importantStyles);
}
export const Pagination = function Pagination({ currentPageIndex, pageSize = 10, numTotal, onChange, style, hideOnSinglePage, dangerouslySetAntdProps, componentId, analyticsEvents, }) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.pagination', false);
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const { pageSizeSelectAriaLabel, pageQuickJumperAriaLabel, ...restDangerouslySetAntdProps } = dangerouslySetAntdProps ?? {};
    const ref = useRef(null);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Pagination,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const onChangeWrapper = useCallback((pageIndex, pageSize) => {
        eventContext.onValueChange(pageIndex);
        onChange(pageIndex, pageSize);
    }, [eventContext, onChange]);
    const { elementRef: paginationRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const mergedRef = useMergeRefs([ref, paginationRef]);
    useEffect(() => {
        if (ref && ref.current) {
            const selectDropdown = ref.current.querySelector(`.${classNamePrefix}-select-selection-search-input`);
            if (selectDropdown) {
                selectDropdown.setAttribute('aria-label', pageSizeSelectAriaLabel ?? 'Select page size');
            }
            const pageQuickJumper = ref.current.querySelector(`.${classNamePrefix}-pagination-options-quick-jumper > input`);
            if (pageQuickJumper) {
                pageQuickJumper.setAttribute('aria-label', pageQuickJumperAriaLabel ?? 'Go to page');
            }
        }
    }, [pageQuickJumperAriaLabel, pageSizeSelectAriaLabel, classNamePrefix]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx("div", { ref: mergedRef, children: _jsx(AntdPagination, { ...addDebugOutlineIfEnabled(), css: getPaginationEmotionStyles(classNamePrefix, theme, useNewShadows, useNewBorderColors), current: currentPageIndex, pageSize: pageSize, responsive: false, total: numTotal, onChange: onChangeWrapper, showSizeChanger: false, showQuickJumper: false, size: "small", style: style, hideOnSinglePage: hideOnSinglePage, ...restDangerouslySetAntdProps, ...eventContext.dataComponentProps }) }) }));
};
export const CursorPagination = function CursorPagination({ onNextPage, onPreviousPage, hasNextPage, hasPreviousPage, nextPageText = 'Next', previousPageText = 'Previous', pageSizeSelect: { options: pageSizeOptions, default: defaultPageSize, getOptionText: getPageSizeOptionText, onChange: onPageSizeChange, ariaLabel = 'Select page size', } = {}, componentId = 'design_system.cursor_pagination', analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], valueHasNoPii, }) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const [pageSizeValue, setPageSizeValue] = useState(defaultPageSize);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const pageSizeEventComponentId = `${componentId}.page_size`;
    const pageSizeEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.LegacySelect,
        componentId: pageSizeEventComponentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const getPageSizeOptionTextDefault = (pageSize) => `${pageSize} / page`;
    return (_jsxs("div", { css: {
            display: 'flex',
            flexDirection: 'row',
            gap: theme.spacing.sm,
            [`.${classNamePrefix}-select-selector::after`]: {
                content: 'none',
            },
        }, ...pageSizeEventContext.dataComponentProps, children: [_jsx(Button, { componentId: `${componentId}.previous_page`, icon: _jsx(ChevronLeftIcon, {}), disabled: !hasPreviousPage, onClick: onPreviousPage, type: "tertiary", children: previousPageText }), _jsx(Button, { componentId: `${componentId}.next_page`, endIcon: _jsx(ChevronRightIcon, {}), disabled: !hasNextPage, onClick: onNextPage, type: "tertiary", children: nextPageText }), pageSizeOptions && (_jsx(LegacySelect, { "aria-label": ariaLabel, value: String(pageSizeValue), css: { width: 120 }, onChange: (pageSize) => {
                    const updatedPageSize = Number(pageSize);
                    onPageSizeChange?.(updatedPageSize);
                    setPageSizeValue(updatedPageSize);
                    // When this usage of LegacySelect is migrated to Select, this call
                    // can be removed in favor of passing a componentId to Select
                    pageSizeEventContext.onValueChange(pageSize);
                }, children: pageSizeOptions.map((pageSize) => (_jsx(LegacySelect.Option, { value: String(pageSize), children: (getPageSizeOptionText || getPageSizeOptionTextDefault)(pageSize) }, pageSize))) }))] }));
};
//# sourceMappingURL=index.js.map