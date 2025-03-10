import { TableFilterInput, TableFilterLayout } from '@databricks/design-system';

export const PromptsListFilters = ({
  searchFilter,
  onSearchFilterChange,
}: {
  searchFilter: string;
  onSearchFilterChange: (searchFilter: string) => void;
}) => {
  return (
    <TableFilterLayout>
      <TableFilterInput
        placeholder="Search prompts"
        componentId="TODO"
        value={searchFilter}
        onChange={(e) => onSearchFilterChange(e.target.value)}
      />
    </TableFilterLayout>
  );
};
