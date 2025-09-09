import { Button, ChevronDownIcon, DropdownMenu, ListBorderIcon } from '@databricks/design-system';
import { defineMessage, FormattedMessage, type MessageDescriptor } from 'react-intl';

import { LoggedModelsTableGroupByMode } from './ExperimentLoggedModelListPageTable.utils';

const GroupByLabels: Record<LoggedModelsTableGroupByMode, MessageDescriptor> = {
  [LoggedModelsTableGroupByMode.RUNS]: defineMessage({
    defaultMessage: 'Source run',
    description: 'Label for the group by runs option in the logged model list page',
  }),
};

export const ExperimentLoggedModelListPageGroupBySelector = ({
  groupBy,
  onChangeGroupBy,
}: {
  groupBy: LoggedModelsTableGroupByMode | undefined;
  onChangeGroupBy?: (groupBy: LoggedModelsTableGroupByMode | undefined) => void;
}) => {
  const currentSelectedLabel = groupBy ? GroupByLabels[groupBy] : null;

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button componentId="mlflow.logged_model.list.group_by" icon={<ListBorderIcon />} endIcon={<ChevronDownIcon />}>
          {currentSelectedLabel ? (
            <FormattedMessage
              defaultMessage="Group by: {currentModeSelected}"
              description="Label for the grouping selector button in the logged model list page when groupin mode is selected"
              values={{ currentModeSelected: <FormattedMessage {...currentSelectedLabel} /> }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="Group by"
              description="Label for the grouping selector button in the logged model list page when no grouping is selected"
            />
          )}
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content>
        <DropdownMenu.CheckboxItem
          checked={!groupBy}
          componentId="mlflow.logged_model.list.group_by.none"
          onClick={() => onChangeGroupBy?.(undefined)}
        >
          <DropdownMenu.ItemIndicator />
          <FormattedMessage
            defaultMessage="None"
            description="Label for the button disabling grouping in the logged model list page"
          />
        </DropdownMenu.CheckboxItem>
        <DropdownMenu.CheckboxItem
          checked={groupBy === LoggedModelsTableGroupByMode.RUNS}
          componentId="mlflow.logged_model.list.group_by.runs"
          onClick={() => onChangeGroupBy?.(LoggedModelsTableGroupByMode.RUNS)}
        >
          <DropdownMenu.ItemIndicator />
          <FormattedMessage {...GroupByLabels[LoggedModelsTableGroupByMode.RUNS]} />
        </DropdownMenu.CheckboxItem>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
