/* eslint-disable @databricks/no-hardcoded-colors */
import { ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { type CSSObject } from '@emotion/react';
import { isObject, truncate } from 'lodash';
import { Fragment, useMemo, useState, useCallback } from 'react';

const MAX_DEPTH = 100;
const INITIAL_EXPAND_DEPTH = 5;
const MAX_STRING_LENGTH = 1000;
const INDENTATION_PER_LEVEL = 16;
const PATH_COLUMN_MAX_WIDTH = '30%';

interface CollapsibleJsonViewerProps {
  data: string;
  initialExpanded?: boolean;
  renderMode?: 'json' | 'table';
}

type JsonColors = {
  key: string;
  string: string;
  number: string;
  boolean: string;
  null: string;
  punctuation: string;
};

function getJsonColors(isDarkMode: boolean, textPrimary: string, textSecondary: string): JsonColors {
  // colors that match Prism's JSON syntax highlighting, this is what users are used to
  // + combination of theme/databricks-duotone-dark.ts and theme/databricks-light.ts
  return {
    key: isDarkMode ? '#5DFAFC' : '#39adb5',
    string: isDarkMode ? '#ffffff' : textPrimary,
    number: isDarkMode ? '#3AACE2' : '#f5871f',
    boolean: isDarkMode ? '#ffffff' : textPrimary,
    null: isDarkMode ? '#ffffff' : textPrimary,
    punctuation: textSecondary,
  };
}

function renderPrimitiveValue(val: unknown, colors: JsonColors): React.ReactNode {
  if (val === null) {
    return <span css={{ color: colors.null }}>null</span>;
  }
  if (val === undefined) {
    return <span css={{ color: colors.null }}>undefined</span>;
  }
  if (typeof val === 'string') {
    const truncated = truncate(val, { length: MAX_STRING_LENGTH });
    return <span css={{ color: colors.string }}>"{truncated}"</span>;
  }
  if (typeof val === 'number') {
    if (Number.isNaN(val)) {
      return <span css={{ color: colors.null }}>NaN</span>;
    }
    if (!Number.isFinite(val)) {
      return <span css={{ color: colors.null }}>{val > 0 ? 'Infinity' : '-Infinity'}</span>;
    }
    return <span css={{ color: colors.number }}>{val}</span>;
  }
  if (typeof val === 'boolean') {
    return <span css={{ color: colors.boolean }}>{String(val)}</span>;
  }
  return <span css={{ color: colors.null }}>{String(val)}</span>;
}

function renderJsonValue(
  val: unknown,
  type: JsonTableRow['type'],
  colors: JsonColors,
  isExpanded: boolean,
): React.ReactNode {
  if (isExpanded) {
    return null;
  }
  if (type === 'array') {
    const arr = val as unknown[];
    const preview = arr.length === 0 ? '[]' : `[...] // ${arr.length} ${arr.length === 1 ? 'item' : 'items'}`;
    return <span css={{ color: colors.punctuation, fontStyle: 'italic' }}>{preview}</span>;
  }
  if (type === 'object') {
    const obj = val as Record<string, unknown>;
    const keys = Object.keys(obj);
    const preview =
      keys.length === 0 ? '{}' : `{...} // ${keys.length} ${keys.length === 1 ? 'property' : 'properties'}`;
    return <span css={{ color: colors.punctuation, fontStyle: 'italic' }}>{preview}</span>;
  }
  return renderPrimitiveValue(val, colors);
}

interface LineInfo {
  lineNumber: number;
  startLine: number;
  endLine: number;
}

function calculateLineNumbers(
  value: unknown,
  pathPrefix: string = '',
  depth: number = 0,
  lineCounter: { current: number } = { current: 1 },
  lineMap: Map<string, LineInfo> = new Map(),
): { lineMap: Map<string, LineInfo>; endLine: number } {
  if (depth > MAX_DEPTH) {
    const line = lineCounter.current++;
    lineMap.set(pathPrefix, { lineNumber: line, startLine: line, endLine: line });
    return { lineMap, endLine: line };
  }

  const isArrayValue = Array.isArray(value);
  const isObjectValue = isObject(value);
  const isPrimitive = !isObjectValue && !isArrayValue;

  if (isPrimitive) {
    const line = lineCounter.current++;
    lineMap.set(pathPrefix, { lineNumber: line, startLine: line, endLine: line });
    return { lineMap, endLine: line };
  }

  const openingLine = lineCounter.current++;
  const entries = isArrayValue
    ? (value as unknown[]).map((v, i) => [String(i), v] as const)
    : Object.entries(value as Record<string, unknown>);

  if (entries.length > 0) {
    entries.forEach(([key, val]) => {
      const childPath = pathPrefix ? `${pathPrefix}.${key}` : key;
      calculateLineNumbers(val, childPath, depth + 1, lineCounter, lineMap);
    });
  }

  const closingLine = lineCounter.current++;
  lineMap.set(pathPrefix, { lineNumber: openingLine, startLine: openingLine, endLine: closingLine });

  return { lineMap, endLine: closingLine };
}

interface JsonNodeProps {
  nodeKey?: string;
  value: unknown;
  depth: number;
  isLast: boolean;
  initialExpanded?: boolean;
  colors: JsonColors;
  theme: any;
  lineMap: Map<string, LineInfo>;
  path: string;
}

function JsonNode({
  nodeKey,
  value,
  depth,
  isLast,
  initialExpanded = false,
  colors,
  theme,
  lineMap,
  path,
}: JsonNodeProps) {
  const [collapsed, setCollapsed] = useState(!initialExpanded && depth > 0);

  const indentSize = theme.spacing.md;
  const isArrayValue = Array.isArray(value);
  const isObjectValue = isObject(value);
  const isPrimitive = !isObjectValue && !isArrayValue;
  const isExpandable = isObjectValue || isArrayValue;

  const lineInfo = lineMap.get(path) || { lineNumber: 1, startLine: 1, endLine: 1 };
  const displayLine = collapsed && lineInfo.endLine !== lineInfo.startLine ? lineInfo.endLine : lineInfo.lineNumber;

  const monoTextStyle: CSSObject = {
    fontFamily: 'monospace',
    fontSize: theme.typography.fontSizeSm,
    lineHeight: theme.typography.lineHeightBase,
  };

  const lineNumberStyle: CSSObject = {
    minWidth: 40,
    paddingRight: theme.spacing.sm,
    textAlign: 'right' as const,
    color: theme.colors.textSecondary,
    userSelect: 'none' as const,
    ...monoTextStyle,
  };

  const lineWrapperStyle: CSSObject = {
    display: 'flex',
    paddingTop: 2,
    paddingBottom: 2,
  };

  if (depth > MAX_DEPTH) {
    return (
      <div css={lineWrapperStyle}>
        <span css={lineNumberStyle}>{displayLine}</span>
        <div
          css={{
            paddingLeft: depth * indentSize,
            color: theme.colors.textSecondary,
            fontStyle: 'italic',
            ...monoTextStyle,
          }}
        >
          [Max depth reached]
        </div>
      </div>
    );
  }

  const getPreview = () => {
    if (isArrayValue) {
      const length = value.length;
      return length === 0 ? '' : ` // ${length} ${length === 1 ? 'item' : 'items'}`;
    }
    if (isObjectValue) {
      const keys = Object.keys(value);
      const count = keys.length;
      return count === 0 ? '' : ` // ${count} ${count === 1 ? 'property' : 'properties'}`;
    }
    return '';
  };

  const brackets = isArrayValue ? { open: '[', close: ']' } : isObjectValue ? { open: '{', close: '}' } : null;

  if (isPrimitive) {
    return (
      <div
        css={{
          display: 'flex',
          paddingTop: 2,
          paddingBottom: 2,
        }}
      >
        <span css={lineNumberStyle}>{displayLine}</span>
        <div
          css={{
            display: 'flex',
            alignItems: 'flex-start',
            paddingLeft: depth * indentSize,
            ...monoTextStyle,
          }}
        >
          {nodeKey !== undefined && (
            <>
              <span css={{ color: colors.key }}>"{nodeKey}"</span>
              <span css={{ color: colors.punctuation, marginLeft: 2, marginRight: theme.spacing.xs }}>:</span>
            </>
          )}
          {renderPrimitiveValue(value, colors)}
          {!isLast && <span css={{ color: colors.punctuation }}>,</span>}
        </div>
      </div>
    );
  }

  const entries = isArrayValue ? value.map((v, i) => [i, v] as const) : Object.entries(value);

  return (
    <div>
      <div
        css={{
          display: 'flex',
          paddingTop: 2,
          paddingBottom: 2,
        }}
      >
        <span css={lineNumberStyle}>{displayLine}</span>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            paddingLeft: depth * indentSize,
            cursor: isExpandable ? 'pointer' : 'default',
            borderRadius: theme.borders.borderRadiusSm,
            marginLeft: -theme.spacing.xs,
            marginRight: -theme.spacing.xs,
            paddingRight: theme.spacing.xs,
            ...monoTextStyle,
          }}
          onClick={() => isExpandable && setCollapsed(!collapsed)}
        >
          {isExpandable ? (
            <span
              css={{
                marginRight: 4,
                marginLeft: theme.spacing.xs,
                display: 'flex',
                alignItems: 'center',
                color: theme.colors.textSecondary,
              }}
            >
              {collapsed ? (
                <ChevronRightIcon css={{ fontSize: theme.spacing.mid }} />
              ) : (
                <ChevronDownIcon css={{ fontSize: theme.spacing.mid }} />
              )}
            </span>
          ) : (
            <span css={{ marginLeft: theme.spacing.xs }} />
          )}
          {nodeKey !== undefined && (
            <>
              <span css={{ color: colors.key }}>"{nodeKey}"</span>
              <span css={{ color: colors.punctuation, marginLeft: 2, marginRight: theme.spacing.xs }}>:</span>
            </>
          )}
          <span css={{ color: colors.punctuation }}>{brackets?.open}</span>
          {collapsed ? (
            <>
              <span css={{ color: colors.punctuation, marginLeft: 2 }}>...</span>
              <span css={{ color: colors.punctuation }}>{brackets?.close}</span>
              <span
                css={{
                  color: theme.colors.textSecondary,
                  marginLeft: theme.spacing.xs,
                  fontStyle: 'italic',
                  ...monoTextStyle,
                }}
              >
                {getPreview()}
              </span>
            </>
          ) : (
            entries.length === 0 && (
              <>
                <span css={{ color: colors.punctuation }}>{brackets?.close}</span>
                {!isLast && <span css={{ color: colors.punctuation }}>,</span>}
              </>
            )
          )}
        </div>
      </div>
      {!collapsed && entries.length > 0 && (
        <>
          {entries.map(([key, val], index) => {
            const childPath = path ? `${path}.${String(key)}` : String(key);
            return (
              <Fragment key={`${depth}-${String(key)}`}>
                <JsonNode
                  nodeKey={isArrayValue ? undefined : String(key)}
                  value={val}
                  depth={depth + 1}
                  isLast={index === entries.length - 1}
                  initialExpanded={depth < INITIAL_EXPAND_DEPTH}
                  colors={colors}
                  theme={theme}
                  lineMap={lineMap}
                  path={childPath}
                />
              </Fragment>
            );
          })}
          <div
            css={{
              display: 'flex',
              paddingTop: 2,
              paddingBottom: 2,
            }}
          >
            <span css={lineNumberStyle}>{lineInfo.endLine}</span>
            <div
              css={{
                paddingLeft: depth * indentSize,
                color: colors.punctuation,
                ...monoTextStyle,
              }}
            >
              {brackets?.close}
              {!isLast && ','}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

interface IdeJsonViewerProps {
  parsedData: unknown;
  initialExpanded: boolean;
  colors: JsonColors;
  theme: any;
}

function IdeJsonViewer({ parsedData, initialExpanded, colors, theme }: IdeJsonViewerProps) {
  const { lineMap } = useMemo(() => calculateLineNumbers(parsedData), [parsedData]);

  return (
    <div css={{ paddingRight: theme.spacing.md * 2 }}>
      <JsonNode
        value={parsedData}
        depth={0}
        isLast
        initialExpanded={initialExpanded}
        colors={colors}
        theme={theme}
        lineMap={lineMap}
        path=""
      />
    </div>
  );
}

interface JsonTableRow {
  id: string;
  key: string;
  value: unknown;
  type: 'string' | 'number' | 'boolean' | 'null' | 'undefined' | 'object' | 'array';
  hasChildren: boolean;
  level: number;
  subRows?: JsonTableRow[];
}

function getValueType(value: unknown): JsonTableRow['type'] {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (Array.isArray(value)) return 'array';
  return typeof value as JsonTableRow['type'];
}

function hasChildren(value: unknown, valueType: JsonTableRow['type']): boolean {
  return (
    (valueType === 'object' && isObject(value) && Object.keys(value as Record<string, unknown>).length > 0) ||
    (valueType === 'array' && Array.isArray(value) && value.length > 0)
  );
}

function transformJsonToTableData(parentKey: string, json: unknown, level: number, parentId: string): JsonTableRow[] {
  if (level > MAX_DEPTH) {
    return [];
  }

  const valueType = getValueType(json);
  const childrenExist = hasChildren(json, valueType);

  if (!childrenExist) {
    const id = parentId ? `${parentId}.${parentKey}` : parentKey;
    return [
      {
        id,
        key: parentKey,
        value: json,
        type: valueType,
        hasChildren: false,
        level,
      },
    ];
  }

  const id = parentId ? `${parentId}.${parentKey}` : parentKey;

  if (Array.isArray(json)) {
    const subRows = json.flatMap((item, index) => transformJsonToTableData(String(index), item, level + 1, id));

    return [
      {
        id,
        key: parentKey,
        value: json,
        type: 'array',
        hasChildren: true,
        level,
        subRows,
      },
    ];
  }

  if (isObject(json)) {
    const entries = Object.entries(json as Record<string, unknown>);
    const subRows = entries.flatMap(([key, value]) => transformJsonToTableData(key, value, level + 1, id));

    return [
      {
        id,
        key: parentKey,
        value: json,
        type: 'object',
        hasChildren: true,
        level,
        subRows,
      },
    ];
  }

  return [];
}

interface JsonTableProps {
  rows: JsonTableRow[];
  colors: JsonColors;
  theme: any;
  initialExpanded: boolean;
}

function JsonTable({ rows, colors, theme, initialExpanded }: JsonTableProps) {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(() => {
    if (!initialExpanded) {
      return new Set();
    }

    const allIds = new Set<string>();
    const collectIds = (items: JsonTableRow[], depth: number) => {
      items.forEach((item) => {
        if (item.hasChildren) {
          allIds.add(item.id);
          if (depth < INITIAL_EXPAND_DEPTH && item.subRows) {
            collectIds(item.subRows, depth + 1);
          }
        }
      });
    };

    collectIds(rows, 0);
    return allIds;
  });

  const toggleRow = useCallback((id: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const pathCellStyle: CSSObject = {
    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
    verticalAlign: 'top',
    maxWidth: PATH_COLUMN_MAX_WIDTH,
    width: 'fit-content',
    minWidth: 0,
    overflow: 'auto',
    whiteSpace: 'nowrap',
    fontFamily: 'monospace',
    fontSize: theme.typography.fontSizeSm,
  };

  const valueCellStyle: CSSObject = {
    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
    verticalAlign: 'top',
    width: 'fit-content',
    maxWidth: '70%',
    minWidth: 0,
    fontFamily: 'monospace',
    fontSize: theme.typography.fontSizeSm,
  };

  const renderRows = (items: JsonTableRow[]): React.ReactNode => {
    return items.map((row) => {
      const isExpanded = expandedRows.has(row.id);

      return (
        <Fragment key={row.id}>
          <tr
            css={{
              borderBottom: `1px solid ${theme.isDarkMode ? theme.colors.grey650 : theme.colors.grey200}`,
            }}
          >
            <td css={pathCellStyle}>
              <div css={{ display: 'flex', alignItems: 'center' }}>
                <div css={{ width: row.level * INDENTATION_PER_LEVEL, flexShrink: 0 }} />
                {row.hasChildren ? (
                  <button
                    onClick={() => toggleRow(row.id)}
                    css={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      padding: 0,
                      marginRight: theme.spacing.xs,
                      display: 'flex',
                      alignItems: 'center',
                      color: theme.colors.textSecondary,
                    }}
                  >
                    {isExpanded ? (
                      <ChevronDownIcon css={{ fontSize: theme.spacing.mid }} />
                    ) : (
                      <ChevronRightIcon css={{ fontSize: theme.spacing.mid }} />
                    )}
                  </button>
                ) : (
                  <div css={{ width: 16, marginRight: theme.spacing.xs, flexShrink: 0 }} />
                )}
                <span css={{ color: colors.key, fontWeight: theme.typography.typographyBoldFontWeight }}>
                  {row.key}
                </span>
              </div>
            </td>
            <td css={valueCellStyle}>{renderJsonValue(row.value, row.type, colors, isExpanded)}</td>
          </tr>
          {isExpanded && row.subRows && row.subRows.length > 0 && renderRows(row.subRows)}
        </Fragment>
      );
    });
  };

  return (
    <div css={{ overflowX: 'auto', width: '100%' }}>
      <table
        css={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: theme.typography.fontSizeSm,
          tableLayout: 'auto',
        }}
      >
        <thead>
          <tr
            css={{
              borderBottom: `1px solid ${theme.isDarkMode ? theme.colors.grey650 : theme.colors.grey200}`,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            <th
              css={{
                padding: `${theme.spacing.sm}px`,
                textAlign: 'left',
                fontWeight: theme.typography.typographyBoldFontWeight,
                color: theme.colors.textSecondary,
                fontSize: theme.typography.fontSizeMd,
                maxWidth: PATH_COLUMN_MAX_WIDTH,
                width: 'fit-content',
                minWidth: 0,
                top: 0,
                backgroundColor: theme.colors.backgroundSecondary,
                zIndex: 1,
              }}
            >
              <FormattedMessage
                defaultMessage="Path"
                description="Table header for the JSON property path column in the collapsible JSON viewer"
              />
            </th>
            <th
              css={{
                padding: `${theme.spacing.sm}px`,
                textAlign: 'left',
                fontWeight: theme.typography.typographyBoldFontWeight,
                color: theme.colors.textSecondary,
                fontSize: theme.typography.fontSizeMd,
                width: 'fit-content',
                minWidth: 0,
                top: 0,
                backgroundColor: theme.colors.backgroundSecondary,
                zIndex: 1,
              }}
            >
              <FormattedMessage
                defaultMessage="Value"
                description="Table header for the JSON property value column in the collapsible JSON viewer"
              />
            </th>
          </tr>
        </thead>
        <tbody>{renderRows(rows)}</tbody>
      </table>
    </div>
  );
}

interface TableJsonViewerProps {
  parsedData: unknown;
  initialExpanded: boolean;
  colors: JsonColors;
  theme: any;
}

function TableJsonViewer({ parsedData, initialExpanded, colors, theme }: TableJsonViewerProps) {
  const tableData = useMemo(() => {
    const json = parsedData;

    if (json === null || json === undefined) {
      return [
        {
          id: 'root',
          key: 'root',
          value: json,
          type: getValueType(json),
          hasChildren: false,
          level: 0,
        },
      ];
    }

    if (Array.isArray(json)) {
      return json.flatMap((item, index) => transformJsonToTableData(String(index), item, 0, ''));
    }

    if (isObject(json)) {
      const entries = Object.entries(json);
      return entries.flatMap(([key, value]) => transformJsonToTableData(key, value, 0, ''));
    }

    return [
      {
        id: 'root',
        key: 'root',
        value: json,
        type: getValueType(json),
        hasChildren: false,
        level: 0,
      },
    ];
  }, [parsedData]);

  return <JsonTable rows={tableData} colors={colors} theme={theme} initialExpanded={initialExpanded} />;
}

export function CollapsibleJsonViewer({
  data,
  initialExpanded = false,
  renderMode = 'json',
}: CollapsibleJsonViewerProps) {
  const { theme } = useDesignSystemTheme();

  const parseResult = useMemo(() => {
    try {
      return { success: true, data: JSON.parse(data) };
    } catch (error) {
      return { success: false, data: undefined };
    }
  }, [data]);

  const colors = useMemo(
    () => getJsonColors(theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary),
    [theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary],
  );

  const isPrimitive = useMemo(() => {
    const data = parseResult.data;
    return (
      data === null ||
      data === undefined ||
      typeof data === 'string' ||
      typeof data === 'number' ||
      typeof data === 'boolean'
    );
  }, [parseResult.data]);

  if (!parseResult.success) {
    return (
      <div
        css={{
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.sm,
          color: theme.colors.textSecondary,
          fontStyle: 'italic',
        }}
      >
        [Invalid JSON]
      </div>
    );
  }

  if (isPrimitive) {
    return (
      <div
        css={{
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.sm,
          fontFamily: 'monospace',
          fontSize: theme.typography.fontSizeSm,
          lineHeight: theme.typography.lineHeightBase,
          borderRadius: theme.borders.borderRadiusSm,
        }}
      >
        {renderPrimitiveValue(parseResult.data, colors)}
      </div>
    );
  }

  return (
    <div
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        padding: theme.spacing.sm,
        position: 'relative',
        borderRadius: theme.borders.borderRadiusSm,
      }}
    >
      {renderMode === 'table' ? (
        <TableJsonViewer
          parsedData={parseResult.data}
          initialExpanded={initialExpanded}
          colors={colors}
          theme={theme}
        />
      ) : (
        <IdeJsonViewer parsedData={parseResult.data} initialExpanded={initialExpanded} colors={colors} theme={theme} />
      )}
    </div>
  );
}
