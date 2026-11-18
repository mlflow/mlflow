/* eslint-disable @databricks/no-hardcoded-colors */
import { ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { CSSObject } from '@emotion/react';
import { isObject } from 'lodash';
import { Fragment, useMemo, useState } from 'react';

import { ModelTraceExplorerJsonRowsViewer } from './ModelTraceExplorerJsonRowsViewer';
import {
  DEFAULT_JSON_ROWS_EXPAND_DEPTH as INITIAL_EXPAND_DEPTH,
  getJsonRows,
  getJsonRowsExpandDepth,
  JSON_ROWS_MAX_DEPTH as MAX_DEPTH,
} from './ModelTraceExplorerJsonRows.utils';
import {
  getJsonColors,
  getJsonContainerPreview,
  renderPrimitiveValue,
  type JsonColors,
} from './ModelTraceExplorerJsonRendering.utils';

export { getJsonColors } from './ModelTraceExplorerJsonRendering.utils';

const LARGE_JSON_MIN_ROWS = 200;
interface CollapsibleJsonViewerProps {
  data: string;
  /** The JSON-compatible value represented by data, when already parsed by the caller. */
  parsedData?: unknown;
  initialExpanded?: boolean;
  /** Applies only to the large-payload row renderer; smaller values keep the legacy view. */
  rowLimit?: number;
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
  lineMap.set(pathPrefix, {
    lineNumber: openingLine,
    startLine: openingLine,
    endLine: closingLine,
  });

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
    if (isArrayValue) return getJsonContainerPreview(true, value.length);
    if (isObjectValue) return getJsonContainerPreview(false, Object.keys(value).length);
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

export function CollapsibleJsonViewer({
  data,
  parsedData,
  initialExpanded = false,
  rowLimit,
}: CollapsibleJsonViewerProps): JSX.Element {
  const { theme } = useDesignSystemTheme();

  const parseResult = useMemo(() => {
    try {
      return { success: true, data: parsedData === undefined ? JSON.parse(data) : parsedData };
    } catch (error) {
      return { success: false, data: undefined };
    }
  }, [data, parsedData]);

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

  const isLargeJson = useMemo(
    () => getJsonRows(parseResult.data, () => true, LARGE_JSON_MIN_ROWS + 1).length > LARGE_JSON_MIN_ROWS,
    [parseResult.data],
  );

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

  if (isLargeJson) {
    return (
      <ModelTraceExplorerJsonRowsViewer
        value={parseResult.data}
        expandDepth={getJsonRowsExpandDepth(data.length)}
        rowLimit={rowLimit}
      />
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
      <IdeJsonViewer parsedData={parseResult.data} initialExpanded={initialExpanded} colors={colors} theme={theme} />
    </div>
  );
}
