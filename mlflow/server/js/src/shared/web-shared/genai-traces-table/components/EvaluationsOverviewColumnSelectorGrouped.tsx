import React, { useMemo, useState, useCallback } from 'react';

import {
  ChevronDownIcon,
  Button,
  ColumnsIcon,
  useDesignSystemTheme,
  DangerIcon,
  Dropdown,
  Input,
  SearchIcon,
  Tree,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { sortGroupedColumns } from '../GenAiTracesTable.utils';
import { TracesTableColumnGroup, TracesTableColumnGroupToLabelMap, type TracesTableColumn } from '../types';

interface Props {
  columns: TracesTableColumn[];
  selectedColumns: TracesTableColumn[];
  toggleColumns: (cols: TracesTableColumn[]) => void;
  setSelectedColumns: (cols: TracesTableColumn[]) => void;
  isMetadataLoading?: boolean;
  metadataError?: Error | null;
}

const getGroupLabel = (group: string): string => {
  return group === TracesTableColumnGroup.INFO
    ? 'Attributes'
    : TracesTableColumnGroupToLabelMap[group as TracesTableColumnGroup];
};

const getGroupKey = (group: string): string => {
  return `GROUP-${group}`;
};

/**
 * Function dissects given string and wraps the
 * searched query with <strong>...</strong> if found. Used for highlighting search.
 */
const createHighlightedNode = (value: string, filterQuery: string) => {
  if (!filterQuery) {
    return value;
  }
  const index = value.toLowerCase().indexOf(filterQuery.toLowerCase());
  const beforeStr = value.substring(0, index);
  const matchStr = value.substring(index, index + filterQuery.length);
  const afterStr = value.substring(index + filterQuery.length);

  return index > -1 ? (
    <span>
      {beforeStr}
      <strong>{matchStr}</strong>
      {afterStr}
    </span>
  ) : (
    value
  );
};

/**
 * Column selector with collapsible tree structure for each column group.
 */
export const EvaluationsOverviewColumnSelectorGrouped: React.FC<React.PropsWithChildren<Props>> = ({
  columns = [],
  selectedColumns = [],
  toggleColumns,
  setSelectedColumns,
  isMetadataLoading = false,
  metadataError,
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const [search, setSearch] = useState('');
  const [dropdownVisible, setDropdownVisible] = useState(false);

  const sortedGroupedColumns = useMemo(() => {
    const sortedColumns = sortGroupedColumns(columns);
    const map: Record<string, TracesTableColumn[]> = {};
    sortedColumns.forEach((col) => {
      const group = col.group ?? TracesTableColumnGroup.INFO;
      if (!map[group]) map[group] = [];
      map[group].push(col);
    });

    return map;
  }, [columns]);

  const filteredGroupedColumns = useMemo(() => {
    if (!search.trim()) return sortedGroupedColumns;

    const needle = search.trim().toLowerCase();
    const out: Record<string, TracesTableColumn[]> = {};

    Object.entries(sortedGroupedColumns).forEach(([group, cols]) => {
      // Check if group name matches
      const groupLabel = getGroupLabel(group);
      const groupMatches = groupLabel.toLowerCase().includes(needle);

      // Check if any columns in the group match
      const hits = cols.filter((c) => c.label.toLowerCase().includes(needle));

      // Include the group if either the group name or any columns match
      if (groupMatches || hits.length) {
        out[group] = groupMatches ? cols : hits;
      }
    });

    return out;
  }, [sortedGroupedColumns, search]);

  // Build tree data structure - memoized to prevent re-creation
  const treeData = useMemo(() => {
    return Object.entries(filteredGroupedColumns).map(([groupName, cols]) => {
      const groupLabel = getGroupLabel(groupName);
      return {
        key: getGroupKey(groupName),
        title: `${groupLabel} (${cols.length})`,
        children: cols.map((col) => ({
          key: col.id,
          title: createHighlightedNode(col.label, search),
        })),
      };
    });
  }, [filteredGroupedColumns, search]);

  // Get all selected column IDs - memoized
  const selectedColumnIds = useMemo(() => selectedColumns.map((col) => col.id), [selectedColumns]);

  // Memoize the check handler to prevent re-creation
  const handleCheck = useCallback(
    (_: any, { node: { key, checked } }: { node: { key: string; checked: boolean } }) => {
      // Check if this is a group node
      if (key.startsWith('GROUP-')) {
        const groupName = key.replace('GROUP-', '');
        const groupColumns = filteredGroupedColumns[groupName] || [];

        if (!checked) {
          // Select all columns in this group
          const columnsToAdd = groupColumns.filter((col) => !selectedColumns.some((c) => c.id === col.id));
          setSelectedColumns([...selectedColumns, ...columnsToAdd]);
        } else {
          // Deselect all columns in this group
          const columnIdsToRemove = groupColumns.map((col) => col.id);
          setSelectedColumns(selectedColumns.filter((col) => !columnIdsToRemove.includes(col.id)));
        }
      } else {
        // Single column toggle
        const column = columns.find((col) => col.id === key);
        if (column) {
          toggleColumns([column]);
        }
      }
    },
    [filteredGroupedColumns, selectedColumns, setSelectedColumns, columns, toggleColumns],
  );

  // Get all group keys for default expansion - memoized
  const defaultExpandedKeys = useMemo(() => {
    return Object.keys(sortedGroupedColumns).map((group) => getGroupKey(group));
  }, [sortedGroupedColumns]);

  const dropdownContent = useMemo(
    () => (
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          width: 400,
          border: `1px solid`,
          borderColor: theme.colors.border,
        }}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            setDropdownVisible(false);
          }
        }}
      >
        {metadataError ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: theme.spacing.xs,
              padding: `${theme.spacing.md}px`,
              color: theme.colors.textValidationDanger,
            }}
            data-testid="filter-dropdown-error"
          >
            <DangerIcon />
            <FormattedMessage
              defaultMessage="Fetching traces failed"
              description="Error message for fetching traces failed"
            />
          </div>
        ) : (
          <>
            <div css={{ padding: theme.spacing.md }}>
              <Input
                componentId="mlflow.traces.column_selector.search"
                value={search}
                prefix={<SearchIcon />}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Search columns',
                  description: 'Placeholder for column selector search input',
                })}
                allowClear
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <div
              css={{
                maxHeight: 15 * 32,
                overflowY: 'scroll',
                overflowX: 'hidden',
                paddingBottom: theme.spacing.md,
                'span[title]': {
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                  overflow: 'hidden',
                },
              }}
            >
              <Tree
                data-testid="column-selector-tree"
                mode="checkable"
                dangerouslySetAntdProps={{
                  checkedKeys: selectedColumnIds,
                  onCheck: handleCheck,
                  // Disable animation for smoother performance
                  motion: null,
                }}
                defaultExpandedKeys={defaultExpandedKeys}
                treeData={treeData}
              />
            </div>
          </>
        )}
      </div>
    ),
    [
      theme,
      metadataError,
      search,
      intl,
      selectedColumnIds,
      handleCheck,
      defaultExpandedKeys,
      treeData,
    ],
  );

  return (
    <Dropdown
      overlay={dropdownContent}
      placement="bottomLeft"
      trigger={['click']}
      visible={isMetadataLoading ? false : dropdownVisible}
      onVisibleChange={setDropdownVisible}
    >
      <Button
        endIcon={<ChevronDownIcon />}
        data-testid="column-selector-button"
        componentId="mlflow.evaluations_review.table_ui.filter_button"
        loading={isMetadataLoading && !metadataError}
      >
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.sm,
            alignItems: 'center',
          }}
        >
          <ColumnsIcon />
          {intl.formatMessage({
            defaultMessage: 'Columns',
            description: 'Evaluation review > evaluations list > filter dropdown button',
          })}
        </div>
      </Button>
    </Dropdown>
  );
};
