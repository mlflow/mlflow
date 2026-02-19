import { TableFilterInput, TableFilterLayout } from '@databricks/design-system';
// eslint-disable-next-line import/no-extraneous-dependencies
import { ModelSearchInputHelpTooltip } from '../../../../model-registry/components/model-list/ModelListFilters';
import type { PromptsListComponentId } from '../PromptsPage';

export const PromptsListFilters = ({
  searchFilter,
  onSearchFilterChange,
  componentId,
}: {
  searchFilter: string;
  onSearchFilterChange: (searchFilter: string) => void;
  componentId: PromptsListComponentId;
}) => {
  return (
    <TableFilterLayout>
      <TableFilterInput
        placeholder="Search prompts by name or tags"
        componentId={componentId}
        value={searchFilter}
        onChange={(e) => onSearchFilterChange(e.target.value)}
        suffix={<ModelSearchInputHelpTooltip exampleEntityName="my-prompt-name" />}
      />
    </TableFilterLayout>
  );
};
