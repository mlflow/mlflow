import React, { useMemo, useState } from 'react';

import {
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxSectionHeader,
  Button,
  ColumnsIcon,
  useDesignSystemTheme,
  DialogComboboxOptionListSearch,
  DangerIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { sortGroupedColumns } from '../GenAiTracesTable.utils';
import { TracesTableColumnGroup, TracesTableColumnGroupToLabelMap, type TracesTableColumn } from '../types';
import { COLUMN_SELECTOR_DROPDOWN_COMPONENT_ID } from '../utils/EvaluationLogging';

interface Props {
  columns: TracesTableColumn[];
  selectedColumns: TracesTableColumn[];
  toggleColumns: (cols: TracesTableColumn[]) => void;
  setSelectedColumns: (cols: TracesTableColumn[]) => void;
  isMetadataLoading?: boolean;
  metadataError?: Error | null;
}

const OPTION_HEIGHT = 32;

const getGroupLabel = (group: string): string => {
  return group === TracesTableColumnGroup.INFO
    ? 'Attributes'
    : TracesTableColumnGroupToLabelMap[group as TracesTableColumnGroup];
};

/**
 * Column selector with section headers for each column‐group.
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

  const handleToggle = (column: TracesTableColumn) => {
    return toggleColumns([column]);
  };

  const handleSelectAllInGroup = (groupColumns: TracesTableColumn[]) => {
    const allSelected = groupColumns.every((col) => selectedColumns.some((c) => c.id === col.id));
    if (allSelected) {
      // If all are selected, deselect all in this group
      const newSelection = selectedColumns.filter((col) => !groupColumns.some((gc) => gc.id === col.id));
      setSelectedColumns(newSelection);
    } else {
      // If not all are selected, select all in this group
      const newSelection = [...selectedColumns];
      groupColumns.forEach((col) => {
        if (!newSelection.some((c) => c.id === col.id)) {
          newSelection.push(col);
        }
      });
      setSelectedColumns(newSelection);
    }
  };

  return (
    <DialogCombobox componentId={COLUMN_SELECTOR_DROPDOWN_COMPONENT_ID} label="Columns" multiSelect>
      <DialogComboboxCustomButtonTriggerWrapper>
        <Button
          endIcon={<ChevronDownIcon />}
          data-testid="column-selector-button"
          componentId="mlflow.evaluations_review.table_ui.filter_button"
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
      </DialogComboboxCustomButtonTriggerWrapper>

      <DialogComboboxContent
        maxHeight={OPTION_HEIGHT * 15.5}
        minWidth={300}
        maxWidth={500}
        loading={isMetadataLoading && !metadataError}
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
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
              {Object.entries(filteredGroupedColumns).map(([groupName, cols]) => (
                <React.Fragment key={groupName}>
                  <DialogComboboxSectionHeader>{getGroupLabel(groupName)}</DialogComboboxSectionHeader>

                  <DialogComboboxOptionListCheckboxItem
                    value={`all-${groupName}`}
                    checked={cols.every((col) => selectedColumns.some((c) => c.id === col.id))}
                    onChange={() => handleSelectAllInGroup(cols)}
                  >
                    {intl.formatMessage(
                      {
                        defaultMessage: 'All {groupLabel}',
                        description: 'Evaluation review > evaluations list > select all columns in group',
                      },
                      { groupLabel: getGroupLabel(groupName) },
                    )}
                  </DialogComboboxOptionListCheckboxItem>

                  {cols.map((col) => (
                    <DialogComboboxOptionListCheckboxItem
                      key={col.id}
                      value={col.label}
                      checked={selectedColumns.some((c) => c.id === col.id)}
                      onChange={() => handleToggle(col)}
                    >
                      {col.label}
                    </DialogComboboxOptionListCheckboxItem>
                  ))}
                </React.Fragment>
              ))}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
        )}
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
