import { css } from '@emotion/react';
import type { Table } from '@tanstack/react-table';
import type { CSSProperties } from 'react';
import React, { useContext } from 'react';

import { getOffsets, genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingState } from '../LoadingState/LoadingState';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { TableContext } from '../TableUI/Table';
import { TableCell } from '../TableUI/TableCell';
import { TableRow } from '../TableUI/TableRow';
import { TableRowAction } from '../TableUI/TableRowAction';
import type { HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface TableSkeletonProps extends HTMLDataAttributes {
  /** Number of rows to render */
  lines?: number;
  /** Seed that deterministically arranges the uneven lines, so that they look like ragged text.
   * If you don't provide this (or give each skeleton the same seed) they will all look the same. */
  seed?: string;
  /** fps for animation. Default is 60 fps. A lower number will use less resources. */
  frameRate?: number;
  /** Style property */
  style?: CSSProperties;
}

const TableSkeletonStyles = {
  container: css({
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
  }),

  cell: css({
    width: '100%',
    height: 8,
    borderRadius: 4,
    background: 'var(--table-skeleton-color)',
    marginTop: 'var(--table-skeleton-row-vertical-margin)',
    marginBottom: 'var(--table-skeleton-row-vertical-margin)',
  }),
};

export const TableSkeleton: React.FC<TableSkeletonProps> = ({
  lines = 1,
  seed = '',
  frameRate = 60,
  style,
  ...rest
}) => {
  const { theme } = useDesignSystemTheme();
  const { size } = useContext(TableContext);
  const widths = getOffsets(seed);

  return (
    <div
      {...rest}
      {...addDebugOutlineIfEnabled()}
      aria-busy={true}
      css={TableSkeletonStyles.container}
      role="status"
      style={{
        ...style,
        // TODO: Pull this from the themes; it's not currently available.
        ['--table-skeleton-color' as any]: theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)',
        ['--table-skeleton-row-vertical-margin' as any]: size === 'small' ? '4px' : '6px',
      }}
    >
      {[...Array(lines)].map((_, idx) => (
        <div
          key={idx}
          css={[
            TableSkeletonStyles.cell,
            genSkeletonAnimatedColor(theme, frameRate),
            { width: `calc(100% - ${widths[idx % widths.length]}px)` },
          ]}
        />
      ))}
    </div>
  );
};

interface TableSkeletonRowsProps<TData> extends WithLoadingState {
  table: Table<TData>;
  actionColumnIds?: string[];
  numRows?: number;
}

interface MinMetaType {
  styles?: CSSProperties;
  width?: number | string;
  numSkeletonLines?: number;
}

export const TableSkeletonRows = <TData, MetaType extends MinMetaType>({
  table,
  actionColumnIds = [],
  numRows = 3,
  loading = true,
  loadingDescription = 'Table skeleton rows',
}: TableSkeletonRowsProps<TData>): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  return (
    <>
      {loading && <LoadingState description={loadingDescription} />}

      {[...Array(numRows).keys()].map((i) => (
        <TableRow key={i}>
          {table.getFlatHeaders().map((header) => {
            const meta = header.column.columnDef.meta as MetaType | undefined;
            return actionColumnIds.includes(header.id) ? (
              <TableRowAction key={`cell-${header.id}-${i}`}>
                <TableSkeleton style={{ width: theme.general.iconSize }} />
              </TableRowAction>
            ) : (
              <TableCell
                key={`cell-${header.id}-${i}`}
                style={meta?.styles ?? (meta?.width !== undefined ? { maxWidth: meta.width } : {})}
              >
                <TableSkeleton seed={`skeleton-${header.id}-${i}`} lines={meta?.numSkeletonLines ?? undefined} />
              </TableCell>
            );
          })}
        </TableRow>
      ))}
    </>
  );
};
