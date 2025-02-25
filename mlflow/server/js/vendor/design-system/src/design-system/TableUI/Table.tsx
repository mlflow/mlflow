import { css } from '@emotion/react';
import classnames from 'classnames';
import type { CSSProperties } from 'react';
import React, { createContext, forwardRef, useImperativeHandle, useMemo, useRef } from 'react';

import tableStyles from './tableStyles';
import {
  DesignSystemEventSuppressInteractionProviderContext,
  DesignSystemEventSuppressInteractionTrueContextValue,
} from '../DesignSystemEventProvider/DesignSystemEventSuppressInteractionProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export const TableContext = createContext<{
  size: 'default' | 'small';
  someRowsSelected?: boolean;
  grid?: boolean;
}>({
  size: 'default',
  grid: false,
});

export interface TableProps extends HTMLDataAttributes {
  size?: 'default' | 'small';
  /** Are any rows currently selected? You must specify this if using `TableRowSelectCell` in your table. */
  someRowsSelected?: boolean;
  /** Style property */
  style?: CSSProperties;
  /** Class name property */
  className?: string;
  /** Content slot for providing a pagination component */
  pagination?: React.ReactNode;
  /** Content slot for providing an Empty component */
  empty?: React.ReactNode;
  /** Child nodes for the table */
  children?: React.ReactNode | React.ReactNode[];
  /** Is this `Table` scrollable? Only use if `Table` is placed within a container of determinate height. */
  scrollable?: boolean;
  /** Adds grid styling to the table (e.g. border around cells and no hover styles) */
  grid?: boolean;
}

export const Table = forwardRef<HTMLDivElement, TableProps>(function Table(
  {
    children,
    size = 'default',
    someRowsSelected,
    style,
    pagination,
    empty,
    className,
    scrollable = false,
    grid = false,
    ...rest
  },
  ref,
): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const tableContentRef = useRef<HTMLDivElement>(null);

  useImperativeHandle(ref, () => tableContentRef.current as HTMLDivElement);

  return (
    <DesignSystemEventSuppressInteractionProviderContext.Provider
      value={DesignSystemEventSuppressInteractionTrueContextValue}
    >
      <TableContext.Provider
        value={useMemo(() => {
          return { size, someRowsSelected, grid };
        }, [size, someRowsSelected, grid])}
      >
        <div
          {...addDebugOutlineIfEnabled()}
          {...rest}
          // This is a performance optimization; we want to statically create the styles for the table,
          // but for the dynamic theme values, we need to use CSS variables.
          // See: https://emotion.sh/docs/best-practices#advanced-css-variables-with-style
          style={{
            ...style,
            ['--table-header-active-color' as any]: theme.colors.actionDefaultTextPress,
            ['colorScheme' as any]: theme.isDarkMode ? 'dark' : undefined,
            ['--table-header-background-color' as any]: theme.colors.backgroundPrimary,
            ['--table-header-focus-color' as any]: theme.colors.actionDefaultTextHover,
            ['--table-header-sort-icon-color' as any]: theme.colors.textSecondary,
            ['--table-header-text-color' as any]: theme.colors.actionDefaultTextDefault,
            ['--table-row-hover' as any]: theme.colors.tableRowHover,
            ['--table-separator-color' as any]: theme.colors.borderDecorative,
            ['--table-resize-handle-color' as any]: theme.colors.borderDecorative,
            ['--table-spacing-md' as any]: `${theme.spacing.md}px`,
            ['--table-spacing-sm' as any]: `${theme.spacing.sm}px`,
            ['--table-spacing-xs' as any]: `${theme.spacing.xs}px`,
          }}
          css={[tableStyles.tableWrapper, css({ minHeight: !empty && pagination ? 150 : 100 })]}
          className={classnames(
            {
              'table-isScrollable': scrollable,
              'table-isGrid': grid,
            },
            className,
          )}
        >
          <div
            role="table"
            ref={tableContentRef}
            css={tableStyles.table}
            // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
            // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
            tabIndex={scrollable ? 0 : -1}
          >
            {children}
            {empty && <div css={{ padding: theme.spacing.lg }}>{empty}</div>}
          </div>
          {!empty && pagination && <div css={tableStyles.paginationContainer}>{pagination}</div>}
        </div>
      </TableContext.Provider>
    </DesignSystemEventSuppressInteractionProviderContext.Provider>
  );
});
