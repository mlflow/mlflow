import { InfoSmallIcon, Popover } from '@databricks/design-system';
import { FormattedMessage, defineMessage, useIntl } from 'react-intl';

const tooltipIntroMessage = defineMessage({
  defaultMessage:
    'To search by tags or by names and tags, use a simplified version{newline}of the SQL {whereBold} clause.',
  description: 'Tooltip to explain how to search MCP servers in the registry',
});

export const MCPSearchInputHelpTooltip = () => {
  const { formatMessage } = useIntl();
  const labelText = formatMessage(tooltipIntroMessage, { newline: ' ', whereBold: 'WHERE' });

  return (
    <Popover.Root componentId="mlflow.mcp_registry.search.help_tooltip">
      <Popover.Trigger
        aria-label={labelText}
        css={{ border: 0, background: 'none', padding: 0, lineHeight: 0, cursor: 'pointer' }}
      >
        <InfoSmallIcon />
      </Popover.Trigger>
      <Popover.Content align="start">
        <div>
          <FormattedMessage {...tooltipIntroMessage} values={{ newline: <br />, whereBold: <b>WHERE</b> }} />
          <br />
          <br />
          <FormattedMessage
            defaultMessage="Examples:"
            description="Text header for examples of MCP server search syntax"
          />
          <br />• tags.my_key = &quot;my_value&quot;
          <br />• name ILIKE &quot;%my-mcp-server%&quot; AND tags.my_key = &quot;my_value&quot;
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};
