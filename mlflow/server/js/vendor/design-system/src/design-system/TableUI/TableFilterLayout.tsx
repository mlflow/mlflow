import { css } from '@emotion/react';
import type { CSSProperties } from 'react';
import React, { forwardRef } from 'react';

import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';

export interface TableFilterLayoutProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
  /** Style property */
  style?: CSSProperties;
  /** Should contain the filter controls to be rendered in this layout. */
  children?: React.ReactNode;
  /** Class name property */
  className?: string;
  /** A container to hold action `Button` elements. */
  actions?: React.ReactNode;
}

export const TableFilterLayout = forwardRef<HTMLDivElement, TableFilterLayoutProps>(function TableFilterLayout(
  { children, style, className, actions, ...rest },
  ref,
) {
  const { theme } = useDesignSystemTheme();

  const tableFilterLayoutStyles = {
    layout: css({
      display: 'flex',
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: 'var(--table-filter-layout-group-margin)',
      columnGap: 'var(--table-filter-layout-group-margin)',
      rowGap: 'var(--table-filter-layout-item-gap)',
      flexWrap: 'wrap',
    }),
    filters: css({
      display: 'flex',
      flexWrap: 'wrap',
      flexDirection: 'row',
      alignItems: 'center',
      gap: 'var(--table-filter-layout-item-gap)',
      marginRight: 'var(--table-filter-layout-group-margin)',
      flex: 1,
    }),
    filterActions: css({
      display: 'flex',
      flexWrap: 'wrap',
      gap: 'var(--table-filter-layout-item-gap)',
      alignSelf: 'flex-start',
    }),
  };

  return (
    <div
      {...rest}
      ref={ref}
      style={{
        ['--table-filter-layout-item-gap' as any]: `${theme.spacing.sm}px`,
        ['--table-filter-layout-group-margin' as any]: `${theme.spacing.md}px`,
        ...style,
      }}
      css={tableFilterLayoutStyles.layout}
      className={className}
    >
      <div css={tableFilterLayoutStyles.filters}>{children}</div>
      {actions && <div css={tableFilterLayoutStyles.filterActions}>{actions}</div>}
    </div>
  );
});
