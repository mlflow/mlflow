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
}

/**
 * A component displaying the searchable column list - implementation.
 */
export const ExperimentViewRunsColumnSelector = React.memo(
  ({
    runsData,
    columnSelectorVisible,
    onChangeColumnSelectorVisible,
    selectedColumns,
  }: ExperimentViewRunsColumnSelectorProps) => {
    const updateUIState = useUpdateExperimentViewUIState();
    const experimentIds = useExperimentIds();
    const [filter, setFilter] = useState('');
    const { theme } = useDesignSystemTheme();

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

    // This memoized value holds the tree structure generated from
    // attributes, params, metrics and tags. Displays only filtered values.
    const treeData = useMemo(() => {
      const result = [];

      const filteredAttributes = findMatching(attributeColumnNames, filter);
      const filteredParams = findMatching(runsData.paramKeyList, filter);
      const filteredMetrics = findMatching(runsData.metricKeyList, filter);
      const filteredTags = findMatching(tagsKeyList, filter);

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
        result.push({
          key: GROUP_KEY_METRICS,
          title: `Metrics (${filteredMetrics.length})`,
          children: filteredMetrics.map((metricKey) => {
            const customColumnDef = customMetricBehaviorDefs[metricKey];
            return {
              key: makeCanonicalSortKey(COLUMN_TYPES.METRICS, metricKey),
              title: createHighlightedNode(customColumnDef?.displayName ?? metricKey, filter),
            };
          }),
        });
      }
      if (filteredParams.length) {
        result.push({
          key: GROUP_KEY_PARAMS,
          title: `Parameters (${filteredParams.length})`,
          children: filteredParams.map((paramKey) => ({
            key: makeCanonicalSortKey(COLUMN_TYPES.PARAMS, paramKey),
            title: createHighlightedNode(paramKey, filter),
          })),
        });
      }
      if (filteredTags.length) {
        result.push({
          key: GROUP_KEY_TAGS,
          title: `Tags (${filteredTags.length})`,
          children: filteredTags.map((tagKey) => ({
            key: makeCanonicalSortKey(COLUMN_TYPES.TAGS, tagKey),
            title: tagKey,
          })),
        });
      }

      return result;
    }, [attributeColumnNames, filter, runsData, tagsKeyList]);

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
        if (isCanonicalSortKeyOfType(key.toString(), GROUP_KEY)) {
          const columnType = extractCanonicalSortKey(key.toString(), GROUP_KEY);
          const canonicalKeysForGroup = canonicalKeyNames[columnType];
          if (canonicalKeysForGroup) {
            toggleGroup(checked, findMatching(canonicalKeysForGroup, filter));
          }
        } else {
          toggleSingleKey(key.toString(), checked);
        }
      },
      [canonicalKeyNames, toggleGroup, toggleSingleKey, filter],
    );

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
