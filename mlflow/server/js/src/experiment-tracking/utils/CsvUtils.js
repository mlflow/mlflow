/**
 * Format a string for insertion into a CSV file.
 */

const csvEscape = (str) => {
    if (str === undefined) {
        return '';
    }
    if (/[,"\r\n]/.test(str)) {
        return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
};

/**
 * Convert a table to a CSV string.
 *
 * @param columns Names of columns
 * @param data Array of rows, each of which are an array of field values
 */
export const tableToCsv = (columns, data) => {
    let csv = '';
    let i;

    for (i = 0; i < columns.length; i++) {
        csv += csvEscape(columns[i]);
        if (i < columns.length - 1) {
            csv += ',';
        }
    }
    csv += '\n';

    for (i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            csv += csvEscape(data[i][j]);
            if (j < data[i].length - 1) {
                csv += ',';
            }
        }
        csv += '\n';
    }

    return csv;
};