import type { TreeDataNode } from '@databricks/design-system';
import { Button, ColumnsIcon, DropdownMenu, Tree, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { compact } from 'lodash';
import { useMemo } from 'react';
import {
  ExperimentLoggedModelListPageKnownColumnGroups,
  ExperimentLoggedModelListPageStaticColumns,
  LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX,
} from './hooks/useExperimentLoggedModelListPageTableColumns';

interface BasicColumnDef {
  colId?: string;
  groupId?: string;
  headerName?: string;
  children?: BasicColumnDef[];
}

const METRIC_AGGREGATE_GROUP_ID = 'all_metrics';

const defaultExpandedTreeGroups = [
  ExperimentLoggedModelListPageKnownColumnGroups.Attributes,
  METRIC_AGGREGATE_GROUP_ID,
];

export const ExperimentLoggedModelListPageColumnSelector = ({
  onUpdateColumns,
  columnVisibility = {},
  columnDefs,
  disabled,
  customTrigger,
}: {
  onUpdateColumns: (columnVisibility: Record<string, boolean>) => void;
  columnVisibility?: Record<string, boolean>;
  columnDefs?: BasicColumnDef[];
  disabled?: boolean;
  customTrigger?: React.ReactNode;
}) => {
  const intl = useIntl();

  // Calculate the tree data for the column selector
  const { leafColumnIds = [], treeData = [] } = useMemo(() => {
    // If there are no column definitions, return an empty tree
    if (!columnDefs) {
      return {};
    }

    // We need to regroup columns so all dataset metric groups are included in another subtree
    const groupedColumnDefinitions: BasicColumnDef[] = [];

    // First, add the attribute column group
    const attributeColumnGroup = columnDefs.find(
      (col) => col.groupId === ExperimentLoggedModelListPageKnownColumnGroups.Attributes,
    );

    if (attributeColumnGroup) {
      groupedColumnDefinitions.push({
        ...attributeColumnGroup,
        // Filter out the static columns
        children: attributeColumnGroup.children?.filter(
          ({ colId }) => colId && !ExperimentLoggedModelListPageStaticColumns.includes(colId),
        ),
      });
    }

    // Next, get all the dataset-grouped metric column groups
    const metricColumnGroups = columnDefs
      .filter((col) => col.groupId?.startsWith(LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX))
      .map((col) => ({
        ...col,
        headerName: col.headerName
          ? `Dataset: ${col.headerName}`
          : intl.formatMessage({
              defaultMessage: 'No dataset',
              description: 'Label for the ungrouped metrics column group in the logged model column selector',
            }),
      }));

    // Aggregate all metric column groups into a single group
    if (metricColumnGroups.length > 0) {
      groupedColumnDefinitions.push({
        groupId: METRIC_AGGREGATE_GROUP_ID,
        headerName: intl.formatMessage({
          defaultMessage: 'Metrics',
          description: 'Label for the metrics column group in the logged model column selector',
        }),
        children: metricColumnGroups,
      });
    }

    // In the end, add the parameter column group
    const paramColumnGroup = columnDefs.find(
      (col) => col.groupId === ExperimentLoggedModelListPageKnownColumnGroups.Params,
    );

    if (paramColumnGroup) {
      groupedColumnDefinitions.push(paramColumnGroup);
    }

    const leafColumnIds: string[] = [];

    // Function for building tree branches recursively
    const buildDuboisTreeBranch = (col: BasicColumnDef): TreeDataNode => {
      if (col.colId) {
        leafColumnIds.push(col.colId);
      }
      return {
        key: col.groupId ?? col.colId ?? '',
        title: col.headerName ?? '',
        children: compact(col.children?.map(buildDuboisTreeBranch) ?? []),
      };
    };

    // Build a tree root for a column groups
    const treeData = compact(groupedColumnDefinitions?.map((col) => buildDuboisTreeBranch(col)) ?? []);

    return {
      leafColumnIds,
      treeData,
    };
  }, [columnDefs, intl]);

  const treeCheckChangeHandler: React.ComponentProps<typeof Tree>['onCheck'] = (checkedKeys) => {
    // Extract key data conforming to unusual antd API
    const keys = 'checked' in checkedKeys ? checkedKeys.checked : checkedKeys;

    // Start with empty visibility map
    const columnVisibility: Record<string, boolean> = {};

    // Go through all leaf columns and set visibility based on the checked keys.
    // We use one-way visibility flag, i.e. use only "false" to hide a column.
    for (const key of leafColumnIds) {
      if (!keys.includes(key)) {
        columnVisibility[key] = false;
      }
    }

    // Call the update handler
    onUpdateColumns(columnVisibility);
  };

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild disabled={disabled}>
        {customTrigger ?? (
          <Button componentId="mlflow.logged_model.list.columns" icon={<ColumnsIcon />} disabled={disabled}>
            <FormattedMessage
              defaultMessage="Columns"
              description="Label for the column selector button in the logged model list page"
            />
          </Button>
        )}
      </DropdownMenu.Trigger>
      <DropdownMenu.Content css={{ maxHeight: 500, paddingRight: 32 }}>
        <Tree
          treeData={treeData}
          mode="checkable"
          showLine
          defaultExpandedKeys={defaultExpandedTreeGroups}
          // By default, check all columns that are visible
          defaultCheckedKeys={leafColumnIds.filter((colId) => columnVisibility[colId] !== false)}
          onCheck={treeCheckChangeHandler}
        />
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
