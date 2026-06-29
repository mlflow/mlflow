import { TableFilterInput, TableFilterLayout } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { ModelSearchInputHelpTooltip } from '../../model-registry/components/model-list/ModelListFilters';

export const MCPServerListFilters = ({
  searchFilter,
  onSearchFilterChange,
  componentId,
}: {
  searchFilter: string;
  onSearchFilterChange: (value: string) => void;
  componentId: string;
}) => {
  const intl = useIntl();
  return (
    <TableFilterLayout>
      <TableFilterInput
        placeholder={intl.formatMessage({
          defaultMessage: 'Search MCP servers by name',
          description: 'Placeholder for MCP server search filter input',
        })}
        componentId={componentId}
        value={searchFilter}
        onChange={(e) => onSearchFilterChange(e.target.value)}
        suffix={<ModelSearchInputHelpTooltip exampleEntityName="my-mcp-server" />}
      />
    </TableFilterLayout>
  );
};
