/// <reference types="react" />
import type { SerializedStyles } from '@emotion/react';
import type { PaginationProps as AntdPaginationProps } from 'antd';
import type { Theme } from '../../theme';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
interface AntdExtraPaginationProps extends AntdPaginationProps {
    pageSizeSelectAriaLabel?: string;
    pageQuickJumperAriaLabel?: string;
}
export interface PaginationProps extends HTMLDataAttributes, DangerouslySetAntdProps<AntdExtraPaginationProps> {
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
export declare function getPaginationEmotionStyles(clsPrefix: string, theme: Theme): SerializedStyles;
export declare const Pagination: React.FC<PaginationProps>;
export interface CursorPaginationProps extends HTMLDataAttributes {
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
export declare const CursorPagination: React.FC<CursorPaginationProps>;
export {};
//# sourceMappingURL=index.d.ts.map