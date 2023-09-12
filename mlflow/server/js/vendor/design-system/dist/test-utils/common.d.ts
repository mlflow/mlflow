/**
 * Used with the `selectEvent` utils for both Enzyme and RTL.
 */
export declare const selectClasses: {
    clear: string;
    item: string;
    list: string;
    open: string;
    option: string;
    removeItem: string;
    selector: string;
};
export interface GetTableRowByCellTextOptions {
    columnHeaderName?: string;
}
export interface TableRows<T> {
    bodyRows: T[];
    headerRow: T;
}
/**
 * @param columns List of column names
 * @param rows List of rows, where each row is a list of cell texts
 * @returns Markdown formatted string representing the data
 *
 * @example
 * // returns the string:
 * //   | Name | Fruit |
 * //   | --- | --- |
 * //   | Alice | Apple |
 * //   | Brady | Banana |
 * createMarkdownTable(['Name', 'Age'], [['Alice', 'Apple'], ['Brady', 'Banana']])
 */
export declare function createMarkdownTable(columns: string[], rows: string[][]): string;
//# sourceMappingURL=common.d.ts.map