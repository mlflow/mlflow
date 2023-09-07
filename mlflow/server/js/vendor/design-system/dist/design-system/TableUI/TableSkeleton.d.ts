import type { Table } from '@tanstack/react-table';
import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export interface TableSkeletonProps extends HTMLDataAttributes {
    /** Number of rows to render */
    lines?: number;
    /** Seed that deterministically arranges the uneven lines, so that they look like ragged text.
     * If you don't provide this (or give each skeleton the same seed) they will all look the same. */
    seed?: string;
    /** Style property */
    style?: CSSProperties;
}
export declare const TableSkeleton: React.FC<TableSkeletonProps>;
interface TableSkeletonRowsProps<TData> {
    table: Table<TData>;
    actionColumnIds?: string[];
    numRows?: number;
}
interface MinMetaType {
    styles?: CSSProperties;
    width?: number | string;
    numSkeletonLines?: number;
}
export declare const TableSkeletonRows: <TData, MetaType extends MinMetaType>({ table, actionColumnIds, numRows, }: TableSkeletonRowsProps<TData>) => React.ReactElement;
export {};
//# sourceMappingURL=TableSkeleton.d.ts.map