import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import type { PaginationProps as AntdPaginationProps } from 'antd';
import { Pagination as AntdPagination } from 'antd';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { Theme } from '../../theme';
import { Button } from '../Button';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronLeftIcon, ChevronRightIcon } from '../Icon';
import { LegacySelect } from '../LegacySelect';
import type {
  AnalyticsEventProps,
  AnalyticsEventValueChangeNoPiiFlagProps,
  DangerouslySetAntdProps,
  HTMLDataAttributes,
} from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface AntdExtraPaginationProps extends AntdPaginationProps {
  pageSizeSelectAriaLabel?: string;
  pageQuickJumperAriaLabel?: string;
}

export interface PaginationProps
  extends HTMLDataAttributes,
    DangerouslySetAntdProps<AntdExtraPaginationProps>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /**
   * The index of the current page. Starts at 1.
   */
  currentPageIndex: number;
  /**
   * The number of results per page.
   */
  pageSize: number;
  /**
   * The total number of results across all pages.
   */
  numTotal: number;
  /**
   * Callback that is triggered when the user navigates to a different page. Recieves the index
   * of the new page and the size of that page.
   */
  onChange: (pageIndex: number, pageSize?: number) => void;
  style?: React.CSSProperties;
  hideOnSinglePage?: boolean;
}

export function getPaginationEmotionStyles(clsPrefix: string, theme: Theme, useNewShadows?: boolean): SerializedStyles {
  const classRoot = `.${clsPrefix}-pagination`;
  const classItem = `.${clsPrefix}-pagination-item`;
  const classLink = `.${clsPrefix}-pagination-item-link`;
  const classActive = `.${clsPrefix}-pagination-item-active`;
  const classEllipsis = `.${clsPrefix}-pagination-item-ellipsis`;
  const classNext = `.${clsPrefix}-pagination-next`;
  const classPrev = `.${clsPrefix}-pagination-prev`;
  const classJumpNext = `.${clsPrefix}-pagination-jump-next`;
  const classJumpPrev = `.${clsPrefix}-pagination-jump-prev`;
  const classSizeChanger = `.${clsPrefix}-pagination-options-size-changer`;
  const classOptions = `.${clsPrefix}-pagination-options`;
  const classDisabled = `.${clsPrefix}-pagination-disabled`;
  const classSelector = `.${clsPrefix}-select-selector`;

  const styles: CSSObject = {
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
      ...(useNewShadows && {
        [`${classSelector}`]: {
          boxShadow: theme.shadows.xs,
        },
      }),
    },
  };

  const importantStyles = importantify(styles);

  return css(importantStyles);
}

export const Pagination: React.FC<PaginationProps> = function Pagination({
  currentPageIndex,
  pageSize = 10,
  numTotal,
  onChange,
  style,
  hideOnSinglePage,
  dangerouslySetAntdProps,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
}: PaginationProps): JSX.Element {
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const { pageSizeSelectAriaLabel, pageQuickJumperAriaLabel, ...restDangerouslySetAntdProps } =
    dangerouslySetAntdProps ?? {};

  const ref = useRef<HTMLDivElement>(null);

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Pagination,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: true,
  });

  const onChangeWrapper = useCallback(
    (pageIndex: number, pageSize?: number) => {
      eventContext.onValueChange(pageIndex);
      onChange(pageIndex, pageSize);
    },
    [eventContext, onChange],
  );

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

  return (
    <DesignSystemAntDConfigProvider>
      <div ref={ref}>
        <AntdPagination
          {...addDebugOutlineIfEnabled()}
          css={getPaginationEmotionStyles(classNamePrefix, theme, useNewShadows)}
          current={currentPageIndex}
          pageSize={pageSize}
          responsive={false}
          total={numTotal}
          onChange={onChangeWrapper}
          showSizeChanger={false}
          showQuickJumper={false}
          size="small"
          style={style}
          hideOnSinglePage={hideOnSinglePage}
          {...restDangerouslySetAntdProps}
          {...eventContext.dataComponentProps}
        />
      </div>
    </DesignSystemAntDConfigProvider>
  );
};

export interface CursorPaginationProps
  extends HTMLDataAttributes,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /** Callback for when the user clicks the next page button. */
  onNextPage: () => void;
  /** Callback for when the user clicks the previous page button. */
  onPreviousPage: () => void;
  /** Whether there is a next page. */
  hasNextPage: boolean;
  /** Whether there is a previous page. */
  hasPreviousPage: boolean;
  /** Text for the next page button. */
  nextPageText?: string;
  /** Text for the previous page button. */
  previousPageText?: string;
  /** Page size options. */
  pageSizeSelect?: {
    /** Page size options. */
    options: number[];
    /** Default page size */
    default: number;
    /** Get page size option text from page size. */
    getOptionText?: (pageSize: number) => string;
    /** onChange handler for page size selector. */
    onChange: (pageSize: number) => void;
    /** Aria label for the page size selector */
    ariaLabel?: string;
  };
}

export const CursorPagination: React.FC<CursorPaginationProps> = function CursorPagination({
  onNextPage,
  onPreviousPage,
  hasNextPage,
  hasPreviousPage,
  nextPageText = 'Next',
  previousPageText = 'Previous',
  pageSizeSelect: {
    options: pageSizeOptions,
    default: defaultPageSize,
    getOptionText: getPageSizeOptionText,
    onChange: onPageSizeChange,
    ariaLabel = 'Select page size',
  } = {},
  componentId = 'design_system.cursor_pagination',
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  valueHasNoPii,
}): JSX.Element {
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

  const getPageSizeOptionTextDefault = (pageSize: number) => `${pageSize} / page`;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        gap: theme.spacing.sm,
        [`.${classNamePrefix}-select-selector::after`]: {
          content: 'none',
        },
      }}
      {...pageSizeEventContext.dataComponentProps}
    >
      <Button
        componentId={`${componentId}.previous_page`}
        icon={<ChevronLeftIcon />}
        disabled={!hasPreviousPage}
        onClick={onPreviousPage}
        type="tertiary"
      >
        {previousPageText}
      </Button>
      <Button
        componentId={`${componentId}.next_page`}
        endIcon={<ChevronRightIcon />}
        disabled={!hasNextPage}
        onClick={onNextPage}
        type="tertiary"
      >
        {nextPageText}
      </Button>
      {pageSizeOptions && (
        <LegacySelect
          aria-label={ariaLabel}
          value={String(pageSizeValue)}
          css={{ width: 120 }}
          onChange={(pageSize) => {
            const updatedPageSize = Number(pageSize);
            onPageSizeChange?.(updatedPageSize);
            setPageSizeValue(updatedPageSize);
            // When this usage of LegacySelect is migrated to Select, this call
            // can be removed in favor of passing a componentId to Select
            pageSizeEventContext.onValueChange(pageSize);
          }}
        >
          {pageSizeOptions.map((pageSize) => (
            <LegacySelect.Option key={pageSize} value={String(pageSize)}>
              {(getPageSizeOptionText || getPageSizeOptionTextDefault)(pageSize)}
            </LegacySelect.Option>
          ))}
        </LegacySelect>
      )}
    </div>
  );
};
