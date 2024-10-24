import React from "react";

interface TableProps {
  rows: Record<string, { formattedName?: string; colSpan?: number }>;
  columns: Record<string, string | JSX.Element>[];
}

export default function Table({ rows, columns }: TableProps) {
  const rowKeys = Object.entries(rows);

  return (
    <div>
      <table>
        <thead>
          <tr>
            {rowKeys.map(([key, { colSpan }]) => (
              <th key={key} colSpan={colSpan || 1}>
                {rows[key].formattedName || key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((column, rowIndex) => (
            <tr key={rowIndex}>
              {rowKeys.map(([key, { colSpan }], colIndex) => (
                <td key={colIndex} colSpan={colSpan || 1}>
                  {column[key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
