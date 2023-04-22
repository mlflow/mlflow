/// <reference types="react" />
import type { TableColumnGroupType as AntDTableColumnGroupType, TableColumnProps as AntDTableColumnProps, TableColumnsType as AntDTableColumnsType, TableColumnType as AntDTableColumnType, TablePaginationConfig as AntDTablePaginationConfig, TableProps as AntDTableProps } from 'antd';
export interface TableProps<RecordType> extends AntDTableProps<RecordType> {
    scrollableInFlexibleContainer?: boolean;
}
export interface TablePaginationConfig extends AntDTablePaginationConfig {
}
export interface TableColumnGroupType<RecordType> extends AntDTableColumnGroupType<RecordType> {
}
export interface TableColumnType<RecordType> extends AntDTableColumnType<RecordType> {
}
export interface TableColumnProps<RecordType> extends AntDTableColumnProps<RecordType> {
}
export interface TableColumnsType<RecordType> extends AntDTableColumnsType<RecordType> {
}
export declare const Table: <T extends object>(props: TableProps<T>) => JSX.Element;
//# sourceMappingURL=Table.d.ts.map