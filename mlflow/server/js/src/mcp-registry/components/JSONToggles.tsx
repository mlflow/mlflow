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

const useJSONToggle = (data: unknown) => {
  const [show, setShow] = useState(false);
  const jsonString = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return { show, setShow, jsonString };
};

export const jsonPreStyles = (theme: ReturnType<typeof useDesignSystemTheme>['theme'], padding = theme.spacing.sm) =>
  ({
    margin: 0,
    padding,
    paddingTop: theme.spacing.xl,
    backgroundColor: theme.colors.backgroundSecondary,
    borderRadius: theme.borders.borderRadiusSm,
    overflow: 'auto' as const,
    fontSize: theme.typography.fontSizeSm,
    maxHeight: 400,
  }) as const;

export const InputSchemaToggle = ({ data }: { data: unknown }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show, setShow, jsonString } = useJSONToggle(data);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.tool_input_schema.toggle"
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={() => setShow(!show)}
        aria-expanded={show}
      >
        <FormattedMessage defaultMessage="Input Schema" description="MCP tool input schema toggle" />
      </Button>
      {show && (
        <div css={{ position: 'relative', marginTop: theme.spacing.xs }}>
          <Tooltip
            componentId="mlflow.mcp_registry.detail.tool_input_schema.copy"
            content={intl.formatMessage({ defaultMessage: 'Copy JSON', description: 'Tooltip for copy JSON button' })}
          >
            <Button
              componentId="mlflow.mcp_registry.detail.tool_input_schema.copy_button"
              size="small"
              icon={<CopyIcon />}
              onClick={() => copyToClipboard(jsonString)}
              css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}
            />
          </Tooltip>
          <pre
            aria-label={intl.formatMessage({
              defaultMessage: 'JSON content',
              description: 'Aria label for JSON code block',
            })}
            css={jsonPreStyles(theme)}
          >
            <code>{jsonString}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

export const OutputSchemaToggle = ({ data }: { data: unknown }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show, setShow, jsonString } = useJSONToggle(data);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.tool_output_schema.toggle"
        type="link"
        icon={show ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={() => setShow(!show)}
        aria-expanded={show}
      >
        <FormattedMessage defaultMessage="Output Schema" description="MCP tool output schema toggle" />
      </Button>
      {show && (
        <div css={{ position: 'relative', marginTop: theme.spacing.xs }}>
          <Tooltip
            componentId="mlflow.mcp_registry.detail.tool_output_schema.copy"
            content={intl.formatMessage({ defaultMessage: 'Copy JSON', description: 'Tooltip for copy JSON button' })}
          >
            <Button
              componentId="mlflow.mcp_registry.detail.tool_output_schema.copy_button"
              size="small"
              icon={<CopyIcon />}
              onClick={() => copyToClipboard(jsonString)}
              css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}
            />
          </Tooltip>
          <pre
            aria-label={intl.formatMessage({
              defaultMessage: 'JSON content',
              description: 'Aria label for JSON code block',
            })}
            css={jsonPreStyles(theme)}
          >
            <code>{jsonString}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

export const RawJSONToggle = ({ serverJson }: { serverJson: ServerJSONPayload }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show: showRaw, setShow: setShowRaw, jsonString } = useJSONToggle(serverJson);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.raw_json.toggle"
        type="link"
        icon={showRaw ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={() => setShowRaw(!showRaw)}
        aria-expanded={showRaw}
      >
        {showRaw ? (
          <FormattedMessage defaultMessage="Hide raw server.json" description="Toggle to hide raw server.json" />
        ) : (
          <FormattedMessage defaultMessage="View raw server.json" description="Toggle to view raw server.json" />
        )}
      </Button>
      {showRaw && (
        <div css={{ position: 'relative', marginTop: theme.spacing.sm }}>
          <Tooltip
            componentId="mlflow.mcp_registry.detail.raw_json.copy"
            content={
              <FormattedMessage defaultMessage="Copy JSON" description="Tooltip for copy raw server.json button" />
            }
          >
            <Button
              componentId="mlflow.mcp_registry.detail.raw_json.copy_button"
              size="small"
              icon={<CopyIcon />}
              onClick={() => copyToClipboard(jsonString)}
              css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}
            />
          </Tooltip>
          <pre
            aria-label={intl.formatMessage({
              defaultMessage: 'JSON content',
              description: 'Aria label for JSON code block',
            })}
            css={jsonPreStyles(theme, theme.spacing.md)}
          >
            <code>{jsonString}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

export const RawToolsJSONToggle = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { show: showRaw, setShow: setShowRaw, jsonString } = useJSONToggle(tools);

  return (
    <div>
      <Button
        componentId="mlflow.mcp_registry.detail.raw_tools_json.toggle"
        type="link"
        icon={showRaw ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={() => setShowRaw(!showRaw)}
        aria-expanded={showRaw}
      >
        {showRaw ? (
          <FormattedMessage defaultMessage="Hide raw tools JSON" description="Toggle to hide raw tools JSON" />
        ) : (
          <FormattedMessage defaultMessage="View raw tools JSON" description="Toggle to view raw tools JSON" />
        )}
      </Button>
      {showRaw && (
        <div css={{ position: 'relative', marginTop: theme.spacing.sm }}>
          <Tooltip
            componentId="mlflow.mcp_registry.detail.raw_tools_json.copy"
            content={
              <FormattedMessage defaultMessage="Copy JSON" description="Tooltip for copy raw tools JSON button" />
            }
          >
            <Button
              componentId="mlflow.mcp_registry.detail.raw_tools_json.copy_button"
              size="small"
              icon={<CopyIcon />}
              onClick={() => copyToClipboard(jsonString)}
              css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}
            />
          </Tooltip>
          <pre
            aria-label={intl.formatMessage({
              defaultMessage: 'JSON content',
              description: 'Aria label for JSON code block',
            })}
            css={jsonPreStyles(theme, theme.spacing.md)}
          >
            <code>{jsonString}</code>
          </pre>
        </div>
      )}
    </div>
  );
};
