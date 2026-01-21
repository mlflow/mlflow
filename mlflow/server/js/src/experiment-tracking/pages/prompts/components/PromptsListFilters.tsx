import { TableFilterInput, TableFilterLayout } from '@databricks/design-system';
// eslint-disable-next-line import/no-extraneous-dependencies
import { ModelSearchInputHelpTooltip } from '../../../../model-registry/components/model-list/ModelListFilters';

export const PromptsListFilters = ({
  searchFilter,
  onSearchFilterChange,
  pageScope,
}: {
  searchFilter: string;
  onSearchFilterChange: (searchFilter: string) => void;
  pageScope: 'global' | 'experiment';
}) => {
  return (
    <TableFilterLayout>
      <TableFilterInput
        placeholder="Search prompts by name or tags"
        componentId={`mlflow.prompts.${pageScope}.list.search`}
        value={searchFilter}
        onChange={(e) => onSearchFilterChange(e.target.value)}
        suffix={<ModelSearchInputHelpTooltip exampleEntityName="my-prompt-name" />}
      />
    </TableFilterLayout>
  );
};
