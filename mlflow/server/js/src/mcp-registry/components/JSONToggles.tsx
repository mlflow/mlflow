import { useMemo, useState } from 'react';
import { Button, ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { MCPTool, ServerJSONPayload } from '../types';
import { LazyJsonRecordEditor } from '../../experiment-tracking/pages/experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor';

const JSONToggle = ({
  data,
  componentId,
  label,
  hideLabel,
  ariaLabel,
  height = '120px',
}: {
  data: unknown;
  componentId: string;
  label: React.ReactNode;
  hideLabel?: React.ReactNode;
  ariaLabel: string;
  height?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [show, setShow] = useState(false);
  const jsonString = useMemo(() => JSON.stringify(data, null, 2), [data]);

  return (
    <div>
      <Button
        componentId={componentId}
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={() => setShow(!show)}
        aria-expanded={show}
      >
        {show && hideLabel ? hideLabel : label}
      </Button>
      {show && (
        <div css={{ marginTop: theme.spacing.xs }}>
          <LazyJsonRecordEditor
            value={jsonString}
            onChange={() => {}}
            readOnly
            height={height}
            maxHeight="400px"
            ariaLabel={ariaLabel}
          />
        </div>
      )}
    </div>
  );
};

export const InputSchemaToggle = ({ data }: { data: unknown }) => (
  <JSONToggle
    data={data}
    componentId="mlflow.mcp_registry.detail.tool_input_schema.toggle"
    label={<FormattedMessage defaultMessage="Input Schema" description="MCP tool input schema toggle" />}
    ariaLabel="Input schema JSON"
  />
);

export const OutputSchemaToggle = ({ data }: { data: unknown }) => (
  <JSONToggle
    data={data}
    componentId="mlflow.mcp_registry.detail.tool_output_schema.toggle"
    label={<FormattedMessage defaultMessage="Output Schema" description="MCP tool output schema toggle" />}
    ariaLabel="Output schema JSON"
  />
);

export const RawJSONToggle = ({ serverJson }: { serverJson: ServerJSONPayload }) => (
  <JSONToggle
    data={serverJson}
    componentId="mlflow.mcp_registry.detail.raw_json.toggle"
    label={<FormattedMessage defaultMessage="View raw server.json" description="Toggle to view raw server.json" />}
    hideLabel={<FormattedMessage defaultMessage="Hide raw server.json" description="Toggle to hide raw server.json" />}
    ariaLabel="Raw server.json"
    height="180px"
  />
);

export const RawToolsJSONToggle = ({ tools }: { tools: MCPTool[] }) => (
  <JSONToggle
    data={tools}
    componentId="mlflow.mcp_registry.detail.raw_tools_json.toggle"
    label={<FormattedMessage defaultMessage="View raw tools JSON" description="Toggle to view raw tools JSON" />}
    hideLabel={<FormattedMessage defaultMessage="Hide raw tools JSON" description="Toggle to hide raw tools JSON" />}
    ariaLabel="Raw tools JSON"
    height="180px"
  />
);
