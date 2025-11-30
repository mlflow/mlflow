import {
  Button,
  InfoSmallIcon,
  Input,
  Popover,
  SearchIcon,
  TableFilterLayout,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { TracesViewControlsActions } from './TracesViewControlsActions';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';

const InputTooltip = ({ baseComponentId }: { baseComponentId: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Popover.Root
      componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewcontrols.tsx_28"
      modal={false}
    >
      <Popover.Trigger asChild>
        <Button
          size="small"
          type="link"
          icon={
            <InfoSmallIcon
              css={{
                svg: { width: 16, height: 16, color: theme.colors.textSecondary },
              }}
            />
          }
          componentId={`${baseComponentId}.traces_table.filter_tooltip`}
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
  rowSelection,
  setRowSelection,
  refreshTraces,
  baseComponentId,
  runUuid,
  traces,
}: {
  experimentIds: string[];
  filter: string;
  onChangeFilter: (newFilter: string) => void;
  rowSelection: { [id: string]: boolean };
  setRowSelection: (newSelection: { [id: string]: boolean }) => void;
  refreshTraces: () => void;
  baseComponentId: string;
  runUuid?: string;
  traces: ModelTraceInfoWithRunName[];
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  // Internal filter value state, used to control the input value
  const [filterValue, setFilterValue] = useState<string | undefined>(filter || undefined);
  const [isEvaluateTracesModalOpen, setEvaluateTracesModalOpen] = useState(false);

  const displayedFilterValue = filterValue ?? filter;

  const selectedRequestIds = Object.entries(rowSelection)
    .filter(([, isSelected]) => isSelected)
    .map(([id]) => id);
  const showActionButtons = selectedRequestIds.length > 0;

  const searchOrDeleteControls = showActionButtons ? (
    <TracesViewControlsActions
      experimentIds={experimentIds}
      rowSelection={rowSelection}
      setRowSelection={setRowSelection}
      refreshTraces={refreshTraces}
      baseComponentId={baseComponentId}
    />
  ) : (
    <TableFilterLayout css={{ marginBottom: 0 }}>
      <Input
        componentId={`${baseComponentId}.traces_table.search_filter`}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search traces',
          description: 'Experiment page > traces view filters > filter string input placeholder',
        })}
        value={displayedFilterValue}
        // Matches runs filter input width
        css={{ width: 430 }}
        onChange={(e) => setFilterValue(e.target.value)}
        prefix={<SearchIcon />}
        suffix={<InputTooltip baseComponentId={baseComponentId} />}
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
    </TableFilterLayout>
  );

  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs }}>
      {/* Search and delete controls */}
      {searchOrDeleteControls}
    </div>
  );
};
