import { TableFilterInput, TableFilterLayout, ToggleButton } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { MCPSearchInputHelpTooltip } from './MCPSearchInputHelpTooltip';

export const MCPServerListFilters = ({
  searchFilter,
  onSearchFilterChange,
  filterActive,
  onFilterActiveChange,
  filterHasEndpoints,
  onFilterHasEndpointsChange,
  componentId,
}: {
  searchFilter: string;
  onSearchFilterChange: (value: string) => void;
  filterActive: boolean;
  onFilterActiveChange: (checked: boolean) => void;
  filterHasEndpoints: boolean;
  onFilterHasEndpointsChange: (checked: boolean) => void;
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
        suffix={<MCPSearchInputHelpTooltip />}
      />
      <ToggleButton
        componentId="mlflow.mcp_registry.filter_active"
        pressed={filterActive}
        onPressedChange={(pressed) => onFilterActiveChange(pressed)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Filter by active status',
          description: 'Aria label for active status filter toggle',
        })}
      >
        <FormattedMessage defaultMessage="Active" description="Filter toggle for active MCP servers" />
      </ToggleButton>
      <ToggleButton
        componentId="mlflow.mcp_registry.filter_has_endpoints"
        pressed={filterHasEndpoints}
        onPressedChange={(pressed) => onFilterHasEndpointsChange(pressed)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Filter by access endpoint availability',
          description: 'Aria label for access endpoint filter toggle',
        })}
      >
        <FormattedMessage
          defaultMessage="Has access endpoint"
          description="Filter toggle for MCP servers with access endpoints"
        />
      </ToggleButton>
    </TableFilterLayout>
  );
};
