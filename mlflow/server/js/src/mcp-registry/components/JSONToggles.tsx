import { useMemo, useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  CopyIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPTool, ServerJSONPayload } from '../types';
import { copyToClipboard } from '../../common/utils/copyToClipboard';
import { jsonPreStyles, overlayButtonStyles } from '../styles';

const useJSONToggle = (data: unknown) => {
  const [show, setShow] = useState(false);
  const jsonString = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return { show, toggle: () => setShow(!show), jsonString };
};

const JSONPreBlock = ({
  jsonString,
  copyTooltip,
  copyButton,
  smSpacing,
  maxHeightArg,
}: {
  jsonString: string;
  copyTooltip: React.ReactNode;
  copyButton: React.ReactNode;
  smSpacing?: boolean;
  maxHeightArg?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ position: 'relative', marginTop: smSpacing ? theme.spacing.sm : theme.spacing.xs }}>
      {copyTooltip}
      <pre
        aria-label={intl.formatMessage({
          defaultMessage: 'JSON content',
          description: 'Aria label for JSON code block',
        })}
        css={jsonPreStyles(theme, maxHeightArg)}
      >
        <code>{jsonString}</code>
      </pre>
    </div>
  );
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
        <JSONPreBlock
          jsonString={jsonString}
          copyTooltip={
            <Tooltip
              componentId="mlflow.mcp_registry.detail.tool_input_schema.copy"
              content={intl.formatMessage({ defaultMessage: 'Copy JSON', description: 'Tooltip for copy JSON button' })}
            >
              <Button
                componentId="mlflow.mcp_registry.detail.tool_input_schema.copy_button"
                size="small"
                icon={<CopyIcon />}
                onClick={() => copyToClipboard(jsonString)}
                css={overlayButtonStyles(theme)}
              />
            </Tooltip>
          }
          copyButton={null}
        />
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
        <JSONPreBlock
          jsonString={jsonString}
          copyTooltip={
            <Tooltip
              componentId="mlflow.mcp_registry.detail.tool_output_schema.copy"
              content={intl.formatMessage({ defaultMessage: 'Copy JSON', description: 'Tooltip for copy JSON button' })}
            >
              <Button
                componentId="mlflow.mcp_registry.detail.tool_output_schema.copy_button"
                size="small"
                icon={<CopyIcon />}
                onClick={() => copyToClipboard(jsonString)}
                css={overlayButtonStyles(theme)}
              />
            </Tooltip>
          }
          copyButton={null}
        />
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
        <JSONPreBlock
          jsonString={jsonString}
          smSpacing
          maxHeightArg={16}
          copyTooltip={
            <Tooltip
              componentId="mlflow.mcp_registry.detail.raw_json.copy"
              content={intl.formatMessage({ defaultMessage: 'Copy JSON', description: 'Tooltip for copy JSON button' })}
            >
              <Button
                componentId="mlflow.mcp_registry.detail.raw_json.copy_button"
                size="small"
                icon={<CopyIcon />}
                onClick={() => copyToClipboard(jsonString)}
                css={overlayButtonStyles(theme)}
              />
            </Tooltip>
          }
          copyButton={null}
        />
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
        <JSONPreBlock
          jsonString={jsonString}
          smSpacing
          maxHeightArg={16}
          copyTooltip={
            <Tooltip
              componentId="mlflow.mcp_registry.detail.raw_tools_json.copy"
              content={intl.formatMessage({ defaultMessage: 'Copy JSON', description: 'Tooltip for copy JSON button' })}
            >
              <Button
                componentId="mlflow.mcp_registry.detail.raw_tools_json.copy_button"
                size="small"
                icon={<CopyIcon />}
                onClick={() => copyToClipboard(jsonString)}
                css={overlayButtonStyles(theme)}
              />
            </Tooltip>
          }
          copyButton={null}
        />
      )}
    </div>
  );
};
