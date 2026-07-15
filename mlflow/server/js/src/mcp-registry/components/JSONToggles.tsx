import { useMemo, useState } from 'react';
import { Button, ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPTool, ServerJSONPayload } from '../types';
import { LazyJsonRecordEditor } from '../../experiment-tracking/pages/experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor';

const useJSONToggle = (data: unknown) => {
  const [show, setShow] = useState(false);
  const jsonString = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return { show, toggle: () => setShow(!show), jsonString };
};

export const InputSchemaToggle = ({ data }: { data: unknown }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show, toggle, jsonString } = useJSONToggle(data);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.tool_input_schema.toggle"
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={toggle}
        aria-expanded={show}
      >
        <FormattedMessage defaultMessage="Input Schema" description="MCP tool input schema toggle" />
      </Button>
      {show && (
        <div css={{ marginTop: theme.spacing.xs }}>
          <LazyJsonRecordEditor
            value={jsonString}
            onChange={() => {}}
            readOnly
            height="120px"
            maxHeight="400px"
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Input schema JSON',
              description: 'Aria label for input schema JSON viewer',
            })}
          />
        </div>
      )}
    </div>
  );
};

export const OutputSchemaToggle = ({ data }: { data: unknown }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show, toggle, jsonString } = useJSONToggle(data);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.tool_output_schema.toggle"
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={toggle}
        aria-expanded={show}
      >
        <FormattedMessage defaultMessage="Output Schema" description="MCP tool output schema toggle" />
      </Button>
      {show && (
        <div css={{ marginTop: theme.spacing.xs }}>
          <LazyJsonRecordEditor
            value={jsonString}
            onChange={() => {}}
            readOnly
            height="120px"
            maxHeight="400px"
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Output schema JSON',
              description: 'Aria label for output schema JSON viewer',
            })}
          />
        </div>
      )}
    </div>
  );
};

export const RawJSONToggle = ({ serverJson }: { serverJson: ServerJSONPayload }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show, toggle, jsonString } = useJSONToggle(serverJson);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.raw_json.toggle"
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={toggle}
        aria-expanded={show}
      >
        {show ? (
          <FormattedMessage defaultMessage="Hide raw server.json" description="Toggle to hide raw server.json" />
        ) : (
          <FormattedMessage defaultMessage="View raw server.json" description="Toggle to view raw server.json" />
        )}
      </Button>
      {show && (
        <div css={{ marginTop: theme.spacing.sm }}>
          <LazyJsonRecordEditor
            value={jsonString}
            onChange={() => {}}
            readOnly
            height="180px"
            maxHeight="400px"
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Raw server.json',
              description: 'Aria label for raw server.json viewer',
            })}
          />
        </div>
      )}
    </div>
  );
};

export const RawToolsJSONToggle = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show, toggle, jsonString } = useJSONToggle(tools);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.raw_tools_json.toggle"
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={toggle}
        aria-expanded={show}
      >
        {show ? (
          <FormattedMessage defaultMessage="Hide raw tools JSON" description="Toggle to hide raw tools JSON" />
        ) : (
          <FormattedMessage defaultMessage="View raw tools JSON" description="Toggle to view raw tools JSON" />
        )}
      </Button>
      {show && (
        <div css={{ marginTop: theme.spacing.sm }}>
          <LazyJsonRecordEditor
            value={jsonString}
            onChange={() => {}}
            readOnly
            height="180px"
            maxHeight="400px"
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Raw tools JSON',
              description: 'Aria label for raw tools JSON viewer',
            })}
          />
        </div>
      )}
    </div>
  );
};
