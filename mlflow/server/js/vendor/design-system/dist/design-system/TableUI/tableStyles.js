import { css } from '@emotion/react';
import { random, times } from 'lodash';
// Class names that can be used to reference children within
// Should not be used outside of design system
// TODO: PE-239 Maybe we could add "dangerous" into the names or make them completely random.
function randomString() {
    return times(20, () => random(35).toString(36)).join('');
}
export const tableClassNames = {
    cell: `js--ds-table-cell-${randomString()}`,
    header: `js--ds-table-header-${randomString()}`,
    row: `js--ds-table-row-${randomString()}`,
};
// We do not want to use `css=` for elements that can appear on the screen more than ~100 times.
// Instead, we define them here and nest the styling in the styles for the table component below.
// For details see: https://emotion.sh/docs/performance
export const repeatingElementsStyles = {
    cell: css({
        display: 'inline-grid',
        position: 'relative',
        flex: 1,
        boxSizing: 'border-box',
        paddingLeft: 'var(--table-spacing-sm)',
        paddingRight: 'var(--table-spacing-sm)',
        wordBreak: 'break-word',
        overflow: 'hidden',
        '& .anticon': {
            verticalAlign: 'text-bottom',
        },
    }),
    header: css({
        fontWeight: 'bold',
        alignItems: 'flex-end',
        display: 'flex',
        overflow: 'hidden',
        '&[aria-sort]': {
            cursor: 'pointer',
            userSelect: 'none',
        },
        '.table-header-text': {
            color: 'var(--table-header-text-color)',
        },
        '.table-header-icon-container': {
            color: 'var(--table-header-sort-icon-color)',
            display: 'none',
        },
        '&[aria-sort]:hover': {
            '.table-header-icon-container, .table-header-text': {
                color: 'var(--table-header-focus-color)',
            },
        },
        '&[aria-sort]:active': {
            '.table-header-icon-container, .table-header-text': {
                color: 'var(--table-header-active-color)',
            },
        },
        '&:hover, &[aria-sort="ascending"], &[aria-sort="descending"]': {
            '.table-header-icon-container': {
                display: 'inline',
            },
        },
    }),
    row: css({
        display: 'flex',
        '&.table-isHeader': {
            '> *': {
                backgroundColor: 'var(--table-header-background-color)',
            },
            '.table-isScrollable &': {
                position: 'sticky',
                top: 0,
                zIndex: 1,
            },
        },
        // Note: Next-sibling selector is necessary for Ant Checkboxes; if we move away
        // from those in the future we would need to adjust these styles.
        '.table-row-select-cell input[type="checkbox"] ~ *': {
            opacity: 'var(--row-checkbox-opacity, 0)',
        },
        '&:not(.table-row-isGrid)&:hover': {
            '&:not(.table-isHeader)': {
                backgroundColor: 'var(--table-row-hover)',
            },
            '.table-row-select-cell input[type="checkbox"] ~ *': {
                opacity: 1,
            },
        },
        '.table-row-select-cell input[type="checkbox"]:focus ~ *': {
            opacity: 1,
        },
        '> *': {
            paddingTop: 'var(--table-row-vertical-padding)',
            paddingBottom: 'var(--table-row-vertical-padding)',
            borderBottom: '1px solid',
            borderColor: 'var(--table-separator-color)',
        },
        '&.table-row-isGrid > *': {
            borderRight: '1px solid',
            borderColor: 'var(--table-separator-color)',
        },
        // Add left border to first cell in grid
        '&.table-row-isGrid > :first-of-type': {
            borderLeft: '1px solid',
            borderColor: 'var(--table-separator-color)',
        },
        // Add top border for first row in cell
        '&.table-row-isGrid.table-isHeader:first-of-type > *': {
            borderTop: '1px solid',
            borderColor: 'var(--table-separator-color)',
        },
    }),
};
export const hideIconButtonActionCellClassName = `hide-icon-button-${randomString()}`;
export const hideIconButtonRowStyles = css({
    [`.${hideIconButtonActionCellClassName} button:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} button:has(> i.fa:only-child),
    .${hideIconButtonActionCellClassName} a:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} a:has(> i.fa:only-child)`]: {
        opacity: 0,
        transition: 'opacity 0.1s ease !important',
    },
    // Show on hover/focus-within
    [`&:hover .${hideIconButtonActionCellClassName} button:has(> span.anticon[role="img"]:only-child),
    &:hover .${hideIconButtonActionCellClassName} button:has(> i.fa:only-child),
    &:hover .${hideIconButtonActionCellClassName} a:has(> span.anticon[role="img"]:only-child),
    &:hover .${hideIconButtonActionCellClassName} a:has(> i.fa:only-child)`]: {
        opacity: 1,
    },
    // Keep visible when actively clicked or when dropdown is open
    [`.${hideIconButtonActionCellClassName} button[aria-expanded="true"]:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} button[aria-expanded="true"]:has(> i.fa:only-child)`]: {
        opacity: 1,
    },
});
// For performance, these styles are defined outside of the component so they are not redefined on every render.
// We're also using CSS Variables rather than any dynamic styles so that the style object remains static.
const tableStyles = {
    tableWrapper: css({
        '&.table-isScrollable': {
            overflow: 'auto',
        },
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        // Inline repeating elements styles for performance reasons
        [`.${tableClassNames.cell}`]: repeatingElementsStyles.cell,
        [`.${tableClassNames.header}`]: repeatingElementsStyles.header,
        [`.${tableClassNames.row}`]: repeatingElementsStyles.row,
    }),
    table: css({
        '.table-isScrollable &': {
            flex: 1,
            overflow: 'auto',
        },
    }),
    headerButtonTarget: css({
        alignItems: 'flex-end',
        display: 'flex',
        overflow: 'hidden',
        width: '100%',
        justifyContent: 'inherit',
        '&:focus': {
            '.table-header-text': {
                color: 'var(--table-header-focus-color)',
            },
            '.table-header-icon-container': {
                color: 'var(--table-header-focus-color)',
                display: 'inline',
            },
        },
        '&:active': {
            '.table-header-icon-container, .table-header-text': {
                color: 'var(--table-header-active-color)',
            },
        },
    }),
    sortHeaderIconOnRight: css({
        marginLeft: 'var(--table-spacing-xs)',
    }),
    sortHeaderIconOnLeft: css({
        marginRight: 'var(--table-spacing-xs)',
    }),
    checkboxCell: css({
        display: 'flex',
        alignItems: 'center',
        flex: 0,
        paddingLeft: 'var(--table-spacing-sm)',
        paddingTop: 0,
        paddingBottom: 0,
        minWidth: 'var(--table-spacing-md)',
        maxWidth: 'var(--table-spacing-md)',
        boxSizing: 'content-box',
    }),
    resizeHandleContainer: css({
        position: 'absolute',
        right: -3,
        top: 'var(--table-spacing-sm)',
        bottom: 'var(--table-spacing-sm)',
        width: 'var(--table-spacing-sm)',
        display: 'flex',
        justifyContent: 'center',
        cursor: 'col-resize',
        userSelect: 'none',
        touchAction: 'none',
        zIndex: 1,
    }),
    resizeHandle: css({
        width: 1,
        background: 'var(--table-resize-handle-color)',
    }),
    paginationContainer: css({
        display: 'flex',
        justifyContent: 'flex-end',
        paddingTop: 'var(--table-spacing-sm)',
        paddingBottom: 'var(--table-spacing-sm)',
    }),
};
export default tableStyles;
//# sourceMappingURL=tableStyles.js.map