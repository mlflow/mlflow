import {
  Button,
  ColumnsIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxTrigger,
  InfoIcon,
  Input,
  Popover,
  SearchIcon,
  TableFilterLayout,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { ExperimentViewTracesTableColumnLabels } from './TracesView.utils';
import { entries } from 'lodash';
import { TracesViewControlsActions } from './TracesViewControlsActions';

const InputTooltip = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <Popover.Root modal={false}>
      <Popover.Trigger asChild>
        <Button
          size="small"
          type="link"
          icon={
            <InfoIcon
              css={{
                svg: { width: 16, height: 16, color: theme.colors.textSecondary },
              }}
            />
          }
          componentId="mlflow.experiment_page.traces_table.filter_tooltip"
        />
      </Popover.Trigger>
      <Popover.Content>
        <Popover.Arrow />
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="Search traces using a simplified version of the SQL {whereBold} clause."
            description="Tooltip string to explain how to search runs from the experiments table"
            values={{ whereBold: <b>WHERE</b> }}
          />
        </Typography.Paragraph>
        <FormattedMessage defaultMessage="Examples:" description="Text header for examples of mlflow search syntax" />
        <ul>
          <li>
            <code>tags.some_tag = "abc"</code>
          </li>
        </ul>
      </Popover.Content>
    </Popover.Root>
  );
};

export const TracesViewControls = ({
  experimentIds,
  filter,
  onChangeFilter,
  hiddenColumns = [],
  disabledColumns = [],
  toggleHiddenColumn,
  rowSelection,
  setRowSelection,
  refreshTraces,
}: {
  experimentIds: string[];
  filter: string;
  onChangeFilter: (newFilter: string) => void;
  hiddenColumns?: string[];
  disabledColumns?: string[];
  toggleHiddenColumn: (columnId: string) => void;
  rowSelection: { [id: string]: boolean };
  setRowSelection: (newSelection: { [id: string]: boolean }) => void;
  refreshTraces: () => void;
}) => {
  const intl = useIntl();

  // Internal filter value state, used to control the input value
  const [filterValue, setFilterValue] = useState<string | undefined>(filter || undefined);

  const allColumnsList = useMemo(() => {
    return entries(ExperimentViewTracesTableColumnLabels)
      .map(([key, label]) => ({
        key,
        label: intl.formatMessage(label),
      }))
      .filter(({ key }) => !disabledColumns.includes(key));
  }, [intl, disabledColumns]);

  const displayedFilterValue = filterValue ?? filter;

  const showActionButtons = Object.values(rowSelection).filter(Boolean).length > 0;

  return showActionButtons ? (
    <TracesViewControlsActions
      experimentIds={experimentIds}
      rowSelection={rowSelection}
      setRowSelection={setRowSelection}
      refreshTraces={refreshTraces}
    />
  ) : (
    <TableFilterLayout css={{ marginBottom: 0 }}>
      <Input
        componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewcontrols.tsx_111"
        placeholder={intl.formatMessage({
          defaultMessage: 'Search traces',
          description: 'Experiment page > traces view filters > filter string input placeholder',
        })}
        value={displayedFilterValue}
        // Matches runs filter input width
        css={{ width: 430 }}
        onChange={(e) => setFilterValue(e.target.value)}
        prefix={<SearchIcon />}
        suffix={<InputTooltip />}
        allowClear
        onClear={() => {
          onChangeFilter('');
          setFilterValue(undefined);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            onChangeFilter(displayedFilterValue);
            setFilterValue(undefined);
          }
        }}
      />
      <DialogCombobox
        componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewcontrols.tsx_135"
        label={
          <div css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <ColumnsIcon />
            <FormattedMessage
              defaultMessage="Columns"
              description="Experiment page > traces table > column selector > title"
            />
          </div>
        }
      >
        <DialogComboboxTrigger
          aria-label={intl.formatMessage({
            defaultMessage: 'Columns',
            description: 'Experiment page > traces table > column selector > title',
          })}
        />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch>
              {allColumnsList.map(({ key, label }) => (
                <DialogComboboxOptionListCheckboxItem
                  key={key}
                  value={key}
                  checked={!hiddenColumns.includes(key)}
                  onChange={() => toggleHiddenColumn(key)}
                >
                  {label}
                </DialogComboboxOptionListCheckboxItem>
              ))}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
    </TableFilterLayout>
  );
};
