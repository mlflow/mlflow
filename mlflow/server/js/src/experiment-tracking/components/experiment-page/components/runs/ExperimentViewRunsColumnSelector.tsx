import {
  Button,
  ChevronDownIcon,
  ColumnsIcon,
  Dropdown,
  Input,
  SearchIcon,
  Tree,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import Utils from '../../../../../common/utils/Utils';
import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../../constants';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { useExperimentIds } from '../../hooks/useExperimentIds';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import {
  extractCanonicalSortKey,
  isCanonicalSortKeyOfType,
  makeCanonicalSortKey,
} from '../../utils/experimentPage.common-utils';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { customMetricBehaviorDefs } from '../../utils/customMetricBehaviorUtils';

/**
 * We need to recreate antd's tree check callback signature since it's not importable
 */
type AntdTreeCheckCallback = { node: { key: string | number; checked: boolean } };

/**
 * Function localizing antd tree inside a DOM element. Used to focusing by keyboard.
 */
const locateAntdTree = (parent: HTMLElement | null): HTMLElement | null =>
  parent?.querySelector('[role="tree"] input') || null;

const GROUP_KEY = 'GROUP';

const GROUP_KEY_ATTRIBUTES = makeCanonicalSortKey(GROUP_KEY, COLUMN_TYPES.ATTRIBUTES);
const GROUP_KEY_PARAMS = makeCanonicalSortKey(GROUP_KEY, COLUMN_TYPES.PARAMS);
const GROUP_KEY_METRICS = makeCanonicalSortKey(GROUP_KEY, COLUMN_TYPES.METRICS);
const GROUP_KEY_TAGS = makeCanonicalSortKey(GROUP_KEY, COLUMN_TYPES.TAGS);

/**
 * Returns all usable attribute columns basing on view mode and enabled flagged features
 */
const getAttributeColumns = (isComparing: boolean) => {
  const result = [
    ATTRIBUTE_COLUMN_LABELS.USER,
    ATTRIBUTE_COLUMN_LABELS.SOURCE,
    ATTRIBUTE_COLUMN_LABELS.VERSION,
    ATTRIBUTE_COLUMN_LABELS.MODELS,
    ATTRIBUTE_COLUMN_LABELS.DESCRIPTION,
  ];

  if (isComparing) {
    result.unshift(ATTRIBUTE_COLUMN_LABELS.EXPERIMENT_NAME);
  }

  result.unshift(ATTRIBUTE_COLUMN_LABELS.DATASET);

  return result;
};

/**
 * Function filters list of string by a given query string.
 */
const findMatching = (values: string[], filterQuery: string) =>
  values.filter((v) => v.toLowerCase().includes(filterQuery.toLowerCase()));

/**
 * Maximum number of items to render per group when no search filter is active.
 * Prevents DOM bloat with thousands of metrics/params.
 */
const MAX_ITEMS_WITHOUT_FILTER = 50;

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
export interface ExperimentViewRunsColumnSelectorProps {
  runsData: ExperimentRunsSelectorResult;
  columnSelectorVisible: boolean;
  onChangeColumnSelectorVisible: (value: boolean) => void;
  selectedColumns: string[];
  onResetColumns: () => void;
}

/**
 * A component displaying the searchable column list - implementation.
 */
export const ExperimentViewRunsColumnSelector = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    runsData,
    columnSelectorVisible,
    onChangeColumnSelectorVisible,
    selectedColumns,
    onResetColumns,
  }: ExperimentViewRunsColumnSelectorProps) => {
    const updateUIState = useUpdateExperimentViewUIState();
    const experimentIds = useExperimentIds();
    const [filter, setFilter] = useState('');
    const { theme } = useDesignSystemTheme();
    const [visibleLimits, setVisibleLimits] = useState<Record<string, number>>({});

    const searchInputRef = useRef<any>(null);
    const scrollableContainerRef = useRef<HTMLDivElement>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);

    // Extract all attribute columns
    const attributeColumnNames = useMemo(() => getAttributeColumns(experimentIds.length > 1), [experimentIds.length]);

    const setCheckedColumns = useCallback(
      (updateFn: (existingCheckedColumns: string[]) => string[]) =>
        updateUIState((facets: ExperimentPageUIState) => {
          const newColumns = updateFn(facets.selectedColumns);
          const uniqueNewColumns = Array.from(new Set(newColumns));
          return { ...facets, selectedColumns: uniqueNewColumns };
        }),
      [updateUIState],
    );

    // Extract unique list of tags
    const tagsKeyList = useMemo(() => Utils.getVisibleTagKeyList(runsData.tagsList), [runsData]);

    // Extract canonical key names for attributes, params, metrics and tags.
    const canonicalKeyNames = useMemo(
      () => ({
        [COLUMN_TYPES.ATTRIBUTES]: attributeColumnNames.map((key) =>
          makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, key),
        ),
        [COLUMN_TYPES.PARAMS]: runsData.paramKeyList.map((key) => makeCanonicalSortKey(COLUMN_TYPES.PARAMS, key)),
        [COLUMN_TYPES.METRICS]: runsData.metricKeyList.map((key) => makeCanonicalSortKey(COLUMN_TYPES.METRICS, key)),
        [COLUMN_TYPES.TAGS]: tagsKeyList.map((key) => makeCanonicalSortKey(COLUMN_TYPES.TAGS, key)),
      }),
      [runsData, attributeColumnNames, tagsKeyList],
    );

    const showMore = useCallback((groupLabel: string) => {
      setVisibleLimits((prev) => ({
        ...prev,
        [groupLabel]: (prev[groupLabel] ?? MAX_ITEMS_WITHOUT_FILTER) + MAX_ITEMS_WITHOUT_FILTER,
      }));
    }, []);

    // This memoized value holds the tree structure generated from
    // attributes, params, metrics and tags. Displays only filtered values.
    // When no search filter is active, caps each group to prevent DOM bloat
    // with thousands of items. Users can click "Show more" to load the next batch.
    const treeData = useMemo(() => {
      const result = [];
      const isFiltering = filter.length > 0;

      const filteredAttributes = findMatching(attributeColumnNames, filter);
      const filteredParams = findMatching(runsData.paramKeyList, filter);
      const filteredMetrics = findMatching(runsData.metricKeyList, filter);
      const filteredTags = findMatching(tagsKeyList, filter);

      const truncateWithShowMore = (
        items: { key: string; title: React.ReactNode }[],
        totalCount: number,
        groupLabel: string,
      ) => {
        if (isFiltering) return items;
        const limit = visibleLimits[groupLabel] ?? MAX_ITEMS_WITHOUT_FILTER;
        if (items.length <= limit) return items;
        const remaining = totalCount - limit;
        const nextBatch = Math.min(remaining, MAX_ITEMS_WITHOUT_FILTER);
        const truncated = items.slice(0, limit);
        truncated.push({
          key: `__show_more_${groupLabel}`,
          title: (
            <span
              data-show-more
              role="button"
              tabIndex={0}
              onClick={(e) => {
                e.stopPropagation();
                showMore(groupLabel);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.stopPropagation();
                  showMore(groupLabel);
                }
              }}
              css={{
                color: theme.colors.actionTertiaryTextDefault,
                cursor: 'pointer',
                '&:hover': { color: theme.colors.actionTertiaryTextHover },
              }}
            >
              Show {nextBatch} more ({remaining} remaining)
            </span>
          ),
        });
        return truncated;
      };

      if (filteredAttributes.length) {
        result.push({
          key: GROUP_KEY_ATTRIBUTES,
          title: `Attributes`,
          children: filteredAttributes.map((attributeKey) => ({
            key: makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, attributeKey),
            title: createHighlightedNode(attributeKey, filter),
          })),
        });
      }
      if (filteredMetrics.length) {
        const metricChildren = filteredMetrics.map((metricKey) => {
          const customColumnDef = customMetricBehaviorDefs[metricKey];
          return {
            key: makeCanonicalSortKey(COLUMN_TYPES.METRICS, metricKey),
            title: createHighlightedNode(customColumnDef?.displayName ?? metricKey, filter),
          };
        });
        result.push({
          key: GROUP_KEY_METRICS,
          title: `Metrics (${filteredMetrics.length})`,
          children: truncateWithShowMore(metricChildren, filteredMetrics.length, 'metrics'),
        });
      }
      if (filteredParams.length) {
        const paramChildren = filteredParams.map((paramKey) => ({
          key: makeCanonicalSortKey(COLUMN_TYPES.PARAMS, paramKey),
          title: createHighlightedNode(paramKey, filter),
        }));
        result.push({
          key: GROUP_KEY_PARAMS,
          title: `Parameters (${filteredParams.length})`,
          children: truncateWithShowMore(paramChildren, filteredParams.length, 'params'),
        });
      }
      if (filteredTags.length) {
        const tagChildren = filteredTags.map((tagKey) => ({
          key: makeCanonicalSortKey(COLUMN_TYPES.TAGS, tagKey),
          title: tagKey,
        }));
        result.push({
          key: GROUP_KEY_TAGS,
          title: `Tags (${filteredTags.length})`,
          children: truncateWithShowMore(tagChildren, filteredTags.length, 'tags'),
        });
      }

      return result;
    }, [attributeColumnNames, filter, runsData, tagsKeyList, theme, visibleLimits, showMore]);

    // This callback toggles entire group of keys
    const toggleGroup = useCallback(
      (isChecked: boolean, keyList: string[]) => {
        if (!isChecked) {
          setCheckedColumns((checked) => [...checked, ...keyList]);
        } else {
          setCheckedColumns((checked) => checked.filter((k) => !keyList.includes(k)));
        }
      },
      [setCheckedColumns],
    );

    // This callback is intended to select/deselect a single key
    const toggleSingleKey = useCallback(
      (key: string, isChecked: boolean) => {
        if (!isChecked) {
          setCheckedColumns((checked) => [...checked, key]);
        } else {
          setCheckedColumns((checked) => checked.filter((k) => k !== key));
        }
      },
      [setCheckedColumns],
    );

    useEffect(() => {
      if (columnSelectorVisible) {
        setFilter('');
        setVisibleLimits({});

        // Let's wait for the next execution frame, then:
        // - restore the dropdown menu scroll position
        // - focus the search input
        // - bring the dropdown into the viewport using scrollIntoView()
        requestAnimationFrame(() => {
          scrollableContainerRef?.current?.scrollTo(0, 0);
          searchInputRef.current?.focus({ preventScroll: true });

          if (buttonRef.current) {
            buttonRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          }
        });
      }
    }, [columnSelectorVisible]);

    const onCheck = useCallback(
      // We need to recreate antd's tree check callback signature
      (_: any, { node: { key, checked } }: AntdTreeCheckCallback) => {
        // Ignore clicks on "Show more" placeholder nodes
        if (key.toString().startsWith('__show_more_')) return;
        if (isCanonicalSortKeyOfType(key.toString(), GROUP_KEY)) {
          const columnType = extractCanonicalSortKey(key.toString(), GROUP_KEY);
          const canonicalKeysForGroup = canonicalKeyNames[columnType];
          if (canonicalKeysForGroup) {
            // When no filter is active, only toggle the currently visible items (paginated view)
            // to avoid selecting thousands of columns at once which would freeze the runs table.
            // Users can click "Show more" to load additional items, then toggle again.
            const isFiltering = filter.length > 0;
            const groupLabelMap: Record<string, string> = {
              [COLUMN_TYPES.METRICS]: 'metrics',
              [COLUMN_TYPES.PARAMS]: 'params',
              [COLUMN_TYPES.TAGS]: 'tags',
            };
            const groupLabel = groupLabelMap[columnType];
            const limit =
              !isFiltering && groupLabel
                ? (visibleLimits[groupLabel] ?? MAX_ITEMS_WITHOUT_FILTER)
                : canonicalKeysForGroup.length;
            const keysToToggle = findMatching(canonicalKeysForGroup, filter).slice(0, limit);
            toggleGroup(checked, keysToToggle);
          }
        } else {
          toggleSingleKey(key.toString(), checked);
        }
      },
      [canonicalKeyNames, toggleGroup, toggleSingleKey, filter, visibleLimits],
    );

    // Clear all selected metrics/params/tags (keeps attribute columns since those are few)
    const clearSelected = useCallback(() => {
      const toRemove = new Set([
        ...canonicalKeyNames[COLUMN_TYPES.METRICS],
        ...canonicalKeyNames[COLUMN_TYPES.PARAMS],
        ...canonicalKeyNames[COLUMN_TYPES.TAGS],
      ]);
      setCheckedColumns((checked) => checked.filter((k) => !toRemove.has(k)));
    }, [canonicalKeyNames, setCheckedColumns]);

    const selectedCount = useMemo(() => {
      const groupKeys = new Set([
        ...canonicalKeyNames[COLUMN_TYPES.METRICS],
        ...canonicalKeyNames[COLUMN_TYPES.PARAMS],
        ...canonicalKeyNames[COLUMN_TYPES.TAGS],
      ]);
      return selectedColumns.filter((k) => groupKeys.has(k)).length;
    }, [canonicalKeyNames, selectedColumns]);

    // This callback moves focus to tree element if down arrow has been pressed
    // when inside search input area.
    const searchInputKeyDown = useCallback<React.KeyboardEventHandler<HTMLInputElement>>((e) => {
      if (e.key === 'ArrowDown') {
        const treeElement = locateAntdTree(scrollableContainerRef.current);

        if (treeElement) {
          treeElement.focus();
        }
      }
    }, []);

    // A JSX block containing the dropdown
    const dropdownContent = (
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          width: 400,
          border: `1px solid`,
          borderColor: theme.colors.border,
          [theme.responsive.mediaQueries.xs]: {
            width: '100vw',
          },
        }}
        onKeyDown={(e) => {
          // Since we're controlling the visibility of the dropdown,
          // we need to handle the escape key to close it.
          if (e.key === 'Escape') {
            onChangeColumnSelectorVisible(false);
            buttonRef.current?.focus();
          }
        }}
      >
        <div css={(theme) => ({ padding: theme.spacing.md })}>
          <Input
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscolumnselector.tsx_300"
            value={filter}
            prefix={<SearchIcon />}
            placeholder="Search columns"
            allowClear
            ref={searchInputRef}
            onChange={(e) => {
              setFilter(e.target.value);
            }}
            onKeyDown={searchInputKeyDown}
          />
          {selectedCount > 0 && (
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginTop: theme.spacing.sm,
                fontSize: theme.typography.fontSizeSm,
                color: theme.colors.textSecondary,
              }}
            >
              <span>{selectedCount} selected</span>
              <Button
                componentId="mlflow_column_selector_clear_selected"
                size="small"
                type="link"
                onClick={clearSelected}
              >
                Clear selected
              </Button>
            </div>
          )}
        </div>
        <div
          ref={scrollableContainerRef}
          css={{
            // Maximum height of 15 elements times 32 pixels as defined in
            // design-system/src/design-system/Tree/Tree.tsx
            maxHeight: 15 * 32,
            overflowY: 'scroll',
            overflowX: 'hidden',
            paddingBottom: theme.spacing.md,
            'span[title]': {
              whiteSpace: 'nowrap',
              textOverflow: 'ellipsis',
              overflow: 'hidden',
            },
            // Hide the checkbox on "Show more" placeholder nodes.
            // Use wildcard class selectors to match regardless of Design System prefix.
            '[class*="treenode"]:has([data-show-more]) [class*="checkbox"]': {
              display: 'none',
            },
            [theme.responsive.mediaQueries.xs]: {
              maxHeight: 'calc(100vh - 100px)',
            },
          }}
        >
          <Tree
            data-testid="column-selector-tree"
            mode="checkable"
            dangerouslySetAntdProps={{
              checkedKeys: selectedColumns,
              onCheck,
            }}
            defaultExpandedKeys={[GROUP_KEY_ATTRIBUTES, GROUP_KEY_PARAMS, GROUP_KEY_METRICS, GROUP_KEY_TAGS]}
            treeData={treeData}
          />
        </div>
        <div
          css={{
            borderTop: `1px solid ${theme.colors.border}`,
            padding: theme.spacing.sm,
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <Button
            componentId="mlflow.experiment_page.runs_table.column_selector.reset_to_defaults"
            type="tertiary"
            data-testid="column-selector-reset"
            onClick={() => {
              onResetColumns();
              onChangeColumnSelectorVisible(false);
            }}
          >
            <FormattedMessage
              defaultMessage="Reset to defaults"
              description="Button in the experiment runs table column selector that resets column visibility, order and width back to defaults"
            />
          </Button>
        </div>
      </div>
    );

    return (
      <Dropdown
        overlay={dropdownContent}
        placement="bottomLeft"
        trigger={['click']}
        visible={columnSelectorVisible}
        onVisibleChange={onChangeColumnSelectorVisible}
      >
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscolumnselector.tsx_315"
          ref={buttonRef}
          style={{ display: 'flex', alignItems: 'center' }}
          data-testid="column-selection-dropdown"
          icon={<ColumnsIcon />}
        >
          <FormattedMessage
            defaultMessage="Columns"
            description="Dropdown text to display columns names that could to be rendered for the experiment runs table"
          />{' '}
          <ChevronDownIcon />
        </Button>
      </Dropdown>
    );
  },
);
