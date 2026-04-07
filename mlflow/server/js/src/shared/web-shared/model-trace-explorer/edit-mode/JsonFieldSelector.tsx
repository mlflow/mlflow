import { useState, useCallback, useMemo } from 'react';
import { Checkbox, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ChevronDownIcon, ChevronRightIcon } from '@databricks/design-system';

interface JsonFieldSelectorProps {
  data: unknown;
  selectedPath: string | null;
  onPathChange: (path: string | null) => void;
  label: string;
}

const buildJsonPath = (segments: (string | number)[]): string =>
  '$' + segments.map((s) => (typeof s === 'number' ? `[${s}]` : `.${s}`)).join('');

const parseJsonPath = (path: string): (string | number)[] => {
  if (!path.startsWith('$')) return [];
  const segments: (string | number)[] = [];
  const rest = path.slice(1);
  const parts = rest.match(/\.([^.[]+)|\[(\d+)\]/g) ?? [];
  for (const part of parts) {
    if (part.startsWith('.')) {
      segments.push(part.slice(1));
    } else {
      const idx = part.match(/\[(\d+)\]/);
      if (idx) segments.push(Number(idx[1]));
    }
  }
  return segments;
};

const truncatePreview = (value: unknown): string => {
  if (typeof value === 'string') {
    const truncated = value.length > 30 ? value.slice(0, 30) + '...' : value;
    return `"${truncated}"`;
  }
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value === null) return 'null';
  if (Array.isArray(value)) return `[${value.length} items]`;
  if (typeof value === 'object') return '{...}';
  return String(value);
};

const isExpandable = (value: unknown): boolean => value !== null && typeof value === 'object';

interface TreeNodeProps {
  name: string | number;
  value: unknown;
  path: (string | number)[];
  selectedSegments: (string | number)[] | null;
  onSelect: (path: (string | number)[]) => void;
  depth: number;
}

const TreeNode = ({ name, value, path, selectedSegments, onSelect, depth }: TreeNodeProps) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const expandable = isExpandable(value);

  const currentPath = [...path, name];
  const isChecked =
    selectedSegments !== null && buildJsonPath(currentPath) === buildJsonPath(selectedSegments);

  const displayName = typeof name === 'number' ? `[${name}]` : name;

  const handleCheckboxChange = () => {
    onSelect(isChecked ? [] : currentPath);
  };

  const entries = useMemo(() => {
    if (!expandable || !expanded) return [];
    if (Array.isArray(value)) return value.map((v, i) => [i, v] as const);
    return Object.entries(value as Record<string, unknown>);
  }, [expandable, expanded, value]);

  return (
    <div>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          paddingLeft: depth * 16,
          paddingTop: 2,
          paddingBottom: 2,
          cursor: expandable ? 'pointer' : 'default',
          '&:hover': { backgroundColor: theme.colors.backgroundSecondary },
        }}
      >
        {expandable ? (
          <span
            onClick={() => setExpanded(!expanded)}
            css={{
              cursor: 'pointer',
              fontSize: 12,
              color: theme.colors.textSecondary,
              width: 16,
              textAlign: 'center',
            }}
          >
            {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          </span>
        ) : (
          <span css={{ width: 16 }} />
        )}
        <Checkbox
          componentId={`json-field-selector.${currentPath.join('.')}`}
          isChecked={isChecked}
          onChange={handleCheckboxChange}
          aria-label={String(displayName)}
        />
        <span
          css={{ color: theme.colors.textPrimary, fontSize: theme.typography.fontSizeSm }}
          onClick={() => expandable && setExpanded(!expanded)}
        >
          {displayName}
        </span>
        {!expanded && (
          <span
            css={{
              color: theme.colors.textSecondary,
              fontSize: theme.typography.fontSizeSm,
              marginLeft: theme.spacing.xs,
            }}
          >
            {truncatePreview(value)}
          </span>
        )}
      </div>
      {expanded &&
        entries.map(([key, childValue]) => (
          <TreeNode
            key={String(key)}
            name={typeof key === 'number' ? key : (key as string)}
            value={childValue}
            path={currentPath}
            selectedSegments={selectedSegments}
            onSelect={onSelect}
            depth={depth + 1}
          />
        ))}
    </div>
  );
};

export const JsonFieldSelector = ({
  data,
  selectedPath,
  onPathChange,
  label,
}: JsonFieldSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const selectedSegments = useMemo(
    () => (selectedPath ? parseJsonPath(selectedPath) : null),
    [selectedPath],
  );

  const handleSelect = useCallback(
    (segments: (string | number)[]) => {
      if (segments.length === 0) {
        onPathChange(null);
      } else {
        onPathChange(buildJsonPath(segments));
      }
    },
    [onPathChange],
  );

  const entries = useMemo(() => {
    if (data === null || data === undefined || typeof data !== 'object') return [];
    if (Array.isArray(data)) return data.map((v, i) => [i, v] as const);
    return Object.entries(data as Record<string, unknown>);
  }, [data]);

  return (
    <div>
      <Typography.Text
        css={{
          fontSize: theme.typography.fontSizeXs,
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          color: theme.colors.textSecondary,
        }}
      >
        {label}
      </Typography.Text>
      <div css={{ marginTop: theme.spacing.xs }}>
        {entries.map(([key, value]) => (
          <TreeNode
            key={String(key)}
            name={typeof key === 'number' ? key : (key as string)}
            value={value}
            path={[]}
            selectedSegments={selectedSegments}
            onSelect={handleSelect}
            depth={0}
          />
        ))}
      </div>
      {selectedPath && (
        <input
          readOnly
          value={selectedPath}
          css={{
            marginTop: theme.spacing.sm,
            width: '100%',
            background: theme.colors.backgroundSecondary,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            color: theme.colors.textSecondary,
            fontSize: theme.typography.fontSizeSm,
          }}
        />
      )}
    </div>
  );
};
