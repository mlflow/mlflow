import { useMemo, useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  CopyIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPTool, ServerJSONPayload } from '../types';
import { copyToClipboard } from '../../common/utils/copyToClipboard';

export const ServerJSONSection = ({ serverJson }: { serverJson: ServerJSONPayload }) => {
  const { theme } = useDesignSystemTheme();
  const packages = serverJson.packages ?? [];
  const remotes = serverJson.remotes ?? [];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {packages.length > 0 && <PackagesSubsection packages={packages} />}
      {remotes.length > 0 && <RemotesSubsection remotes={remotes} />}
      <RawJSONToggle serverJson={serverJson} />
    </div>
  );
};

export const ToolsSection = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <ToolsSubsection tools={tools} />
      <RawToolsJSONToggle tools={tools} />
    </div>
  );
};

const INITIAL_VISIBLE_PACKAGES = 5;

const PackagesSubsection = ({ packages }: { packages: NonNullable<ServerJSONPayload['packages']> }) => {
  const { theme } = useDesignSystemTheme();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const visiblePackages = showAll ? packages : packages.slice(0, INITIAL_VISIBLE_PACKAGES);
  const hiddenCount = packages.length - INITIAL_VISIBLE_PACKAGES;

  return (
    <div>
      <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Packages ({count})"
          description="MCP server version detail packages subsection heading"
          values={{ count: packages.length }}
        />
      </Typography.Text>
      <div
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        {visiblePackages.map((pkg, index) => (
          <PackageRow
            key={`${pkg.registryType}-${pkg.identifier}`}
            pkg={pkg}
            expanded={expandedIndex === index}
            onToggle={() => setExpandedIndex(expandedIndex === index ? null : index)}
            showTopBorder={index > 0}
          />
        ))}
        {hiddenCount > 0 && (
          <div
            css={{
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
              padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
              textAlign: 'center',
            }}
          >
            <Button
              componentId="mlflow.mcp_registry.detail.toggle_packages"
              type="link"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? (
                <FormattedMessage
                  defaultMessage="Show less"
                  description="MCP server version detail show less packages button"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Show {count} more"
                  description="MCP server version detail show more packages button"
                  values={{ count: hiddenCount }}
                />
              )}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

const PackageRow = ({
  pkg,
  expanded,
  onToggle,
  showTopBorder,
}: {
  pkg: NonNullable<ServerJSONPayload['packages']>[number];
  expanded: boolean;
  onToggle: () => void;
  showTopBorder: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const allEnvVars = pkg.environmentVariables ?? [];
  const runtimeHint = pkg.runtimeHint;
  const transportLabel = [pkg.transport?.type, runtimeHint].filter(Boolean).join(' · ');

  return (
    <div
      css={{
        borderTop: showTopBorder ? `1px solid ${theme.colors.borderDecorative}` : 'none',
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-label={intl.formatMessage(
          {
            defaultMessage: '{action} package {identifier}',
            description: 'Aria label for expanding/collapsing a package row',
          },
          {
            action: expanded ? 'Collapse' : 'Expand',
            identifier: pkg.identifier,
          },
        )}
        css={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          gap: theme.spacing.sm,
          textAlign: 'left',
          '&:hover': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
          },
        }}
      >
        <div
          css={{
            flexShrink: 0,
            width: theme.spacing.md,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        </div>
        <Tag componentId="mlflow.mcp_registry.detail.package_registry_tag" color="turquoise" css={{ flexShrink: 0 }}>
          {pkg.registryType}
        </Tag>
        <Typography.Text
          color="secondary"
          css={{
            fontFamily: 'monospace',
            fontSize: theme.typography.fontSizeSm,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
            minWidth: 0,
          }}
        >
          {pkg.identifier}
        </Typography.Text>
      </button>

      {expanded && (
        <div
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.md}px ${theme.spacing.md}px`,
            paddingLeft: theme.spacing.md + theme.spacing.md + theme.spacing.sm,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Typography.Text css={{ fontFamily: 'monospace' }}>{pkg.identifier}</Typography.Text>
            <Tooltip
              componentId="mlflow.mcp_registry.detail.copy_package_identifier"
              content={intl.formatMessage({
                defaultMessage: 'Copy identifier',
                description: 'Tooltip for copy package identifier button',
              })}
            >
              <Button
                componentId="mlflow.mcp_registry.detail.copy_package_identifier_button"
                size="small"
                icon={<CopyIcon />}
                onClick={(e) => {
                  e.stopPropagation();
                  copyToClipboard(pkg.identifier);
                }}
              />
            </Tooltip>
          </div>

          {pkg.version && (
            <div css={{ display: 'flex', alignItems: 'baseline', gap: theme.spacing.xs }}>
              <Typography.Text bold size="sm">
                <FormattedMessage defaultMessage="Version:" description="MCP server package version label" />
              </Typography.Text>
              <Typography.Text size="sm">{pkg.version}</Typography.Text>
            </div>
          )}

          {pkg.transport?.type && (
            <div css={{ display: 'flex', alignItems: 'baseline', gap: theme.spacing.xs }}>
              <Typography.Text bold size="sm">
                <FormattedMessage defaultMessage="Transport:" description="MCP server package transport label" />
              </Typography.Text>
              <Typography.Text size="sm">{transportLabel}</Typography.Text>
            </div>
          )}

          {allEnvVars.length > 0 && <EnvVarList envVars={allEnvVars} />}
        </div>
      )}
    </div>
  );
};

const INITIAL_VISIBLE_ENV_VARS = 5;

const EnvVarList = ({
  envVars,
}: {
  envVars: NonNullable<ServerJSONPayload['packages']>[number]['environmentVariables'];
}) => {
  const { theme } = useDesignSystemTheme();
  const vars = envVars ?? [];
  const [showAll, setShowAll] = useState(false);
  const visibleVars = showAll ? vars : vars.slice(0, INITIAL_VISIBLE_ENV_VARS);
  const hiddenCount = vars.length - INITIAL_VISIBLE_ENV_VARS;

  return (
    <div>
      <Typography.Text bold size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Environment Variables ({count})"
          description="MCP server package environment variables heading with count"
          values={{ count: vars.length }}
        />
      </Typography.Text>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.borders.borderRadiusSm,
          overflow: 'hidden',
        }}
      >
        {visibleVars.map((envVar, i) => (
          <div
            key={envVar.name}
            css={{
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              borderTop: i > 0 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
            }}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <Typography.Text bold size="sm" css={{ fontFamily: 'monospace' }}>
                {envVar.name}
              </Typography.Text>
              {envVar.isRequired && (
                <Tag componentId="mlflow.mcp_registry.detail.env_var_required" color="lemon">
                  <FormattedMessage defaultMessage="required" description="MCP server package env var required badge" />
                </Tag>
              )}
              {envVar.isSecret && (
                <Tag componentId="mlflow.mcp_registry.detail.env_var_secret" color="coral">
                  <FormattedMessage defaultMessage="secret" description="MCP server package env var secret badge" />
                </Tag>
              )}
            </div>
            {envVar.description && (
              <Typography.Text color="secondary" size="sm">
                {envVar.description}
              </Typography.Text>
            )}
          </div>
        ))}
        {hiddenCount > 0 && (
          <div
            css={{
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              textAlign: 'center',
            }}
          >
            <Button
              componentId="mlflow.mcp_registry.detail.toggle_env_vars"
              type="link"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? (
                <FormattedMessage
                  defaultMessage="Show less"
                  description="MCP server package show less env vars button"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Show {count} more"
                  description="MCP server package show more env vars button"
                  values={{ count: hiddenCount }}
                />
              )}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

const RemotesSubsection = ({ remotes }: { remotes: NonNullable<ServerJSONPayload['remotes']> }) => {
  const { theme } = useDesignSystemTheme();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  return (
    <div>
      <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Remotes ({count})"
          description="MCP server version detail remotes subsection heading"
          values={{ count: remotes.length }}
        />
      </Typography.Text>
      <div
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        {remotes.map((remote, index) => (
          <RemoteRow
            key={`${remote.type}-${remote.url ?? index}`}
            remote={remote}
            expanded={expandedIndex === index}
            onToggle={() => setExpandedIndex(expandedIndex === index ? null : index)}
            showTopBorder={index > 0}
          />
        ))}
      </div>
    </div>
  );
};

const RemoteRow = ({
  remote,
  expanded,
  onToggle,
  showTopBorder,
}: {
  remote: NonNullable<ServerJSONPayload['remotes']>[number];
  expanded: boolean;
  onToggle: () => void;
  showTopBorder: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ borderTop: showTopBorder ? `1px solid ${theme.colors.borderDecorative}` : 'none' }}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-label={intl.formatMessage(
          {
            defaultMessage: '{action} remote {url}',
            description: 'Aria label for expanding/collapsing a remote row',
          },
          { action: expanded ? 'Collapse' : 'Expand', url: remote.url ?? remote.type },
        )}
        css={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          gap: theme.spacing.sm,
          textAlign: 'left',
          '&:hover': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
          },
        }}
      >
        <div
          css={{
            flexShrink: 0,
            width: theme.spacing.md,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        </div>
        <Tag componentId="mlflow.mcp_registry.detail.remote_transport_tag" color="charcoal" css={{ flexShrink: 0 }}>
          {remote.type}
        </Tag>
        {remote.url && (
          <Typography.Text
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
              minWidth: 0,
            }}
          >
            {remote.url}
          </Typography.Text>
        )}
      </button>

      {expanded && (
        <div
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.md}px ${theme.spacing.md}px`,
            paddingLeft: theme.spacing.md + theme.spacing.md + theme.spacing.sm,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {remote.url && (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <Typography.Text css={{ fontFamily: 'monospace' }}>{remote.url}</Typography.Text>
              <Tooltip
                componentId="mlflow.mcp_registry.detail.copy_remote_url"
                content={intl.formatMessage({
                  defaultMessage: 'Copy URL',
                  description: 'Tooltip for copy remote URL button',
                })}
              >
                <Button
                  componentId="mlflow.mcp_registry.detail.copy_remote_url_button"
                  size="small"
                  icon={<CopyIcon />}
                  onClick={(e) => {
                    e.stopPropagation();
                    copyToClipboard(remote.url ?? '');
                  }}
                />
              </Tooltip>
            </div>
          )}

          {remote.headers && remote.headers.length > 0 && (
            <div>
              <Typography.Text bold size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="Headers ({count})"
                  description="MCP server remote headers heading with count"
                  values={{ count: remote.headers.length }}
                />
              </Typography.Text>
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  border: `1px solid ${theme.colors.borderDecorative}`,
                  borderRadius: theme.borders.borderRadiusSm,
                  overflow: 'hidden',
                }}
              >
                {remote.headers.map((header, i) => (
                  <div
                    key={header.name}
                    css={{
                      padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                      borderTop: i > 0 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                    }}
                  >
                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                      <Typography.Text bold size="sm" css={{ fontFamily: 'monospace' }}>
                        {header.name}
                      </Typography.Text>
                      {header.isRequired && (
                        <Tag componentId="mlflow.mcp_registry.detail.header_required" color="lemon">
                          <FormattedMessage defaultMessage="required" description="Header required badge" />
                        </Tag>
                      )}
                      {header.isSecret && (
                        <Tag componentId="mlflow.mcp_registry.detail.header_secret" color="coral">
                          <FormattedMessage defaultMessage="secret" description="Header secret badge" />
                        </Tag>
                      )}
                    </div>
                    {header.description && (
                      <Typography.Text color="secondary" size="sm">
                        {header.description}
                      </Typography.Text>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {remote.variables && Object.keys(remote.variables).length > 0 && (
            <div>
              <Typography.Text bold size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="URL Variables"
                  description="MCP server remote URL variables heading"
                />
              </Typography.Text>
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  border: `1px solid ${theme.colors.borderDecorative}`,
                  borderRadius: theme.borders.borderRadiusSm,
                  overflow: 'hidden',
                }}
              >
                {Object.entries(remote.variables).map(([name, variable], i) => {
                  const v =
                    typeof variable === 'object' && variable !== null ? (variable as Record<string, unknown>) : {};
                  return (
                    <div
                      key={name}
                      css={{
                        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                        borderTop: i > 0 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                      }}
                    >
                      <Typography.Text bold size="sm" css={{ fontFamily: 'monospace' }}>
                        {`{${name}}`}
                      </Typography.Text>
                      {v['description'] && (
                        <Typography.Text color="secondary" size="sm" css={{ display: 'block' }}>
                          {String(v['description'])}
                        </Typography.Text>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const INITIAL_VISIBLE_TOOLS = 10;

const ToolsSubsection = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const visibleTools = showAll ? tools : tools.slice(0, INITIAL_VISIBLE_TOOLS);
  const hiddenCount = tools.length - INITIAL_VISIBLE_TOOLS;

  return (
    <div>
      <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Tools ({count})"
          description="MCP server version detail tools subsection heading"
          values={{ count: tools.length }}
        />
      </Typography.Text>
      <div
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        {visibleTools.map((tool, index) => (
          <div
            key={tool.name}
            css={{
              borderTop: index > 0 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
            }}
          >
            <button
              type="button"
              onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
              aria-expanded={expandedIndex === index}
              aria-label={intl.formatMessage(
                {
                  defaultMessage: '{action} tool {name}',
                  description: 'Aria label for expanding/collapsing a tool row',
                },
                {
                  action: expandedIndex === index ? 'Collapse' : 'Expand',
                  name: tool.name,
                },
              )}
              css={{
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                gap: theme.spacing.sm,
                textAlign: 'left',
                '&:hover': {
                  backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                },
              }}
            >
              <div
                css={{
                  flexShrink: 0,
                  width: theme.spacing.md,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {expandedIndex === index ? <ChevronDownIcon /> : <ChevronRightIcon />}
              </div>
              <Tag componentId="mlflow.mcp_registry.detail.tool_name_tag" color="turquoise" css={{ flexShrink: 0 }}>
                {tool.name}
              </Tag>
              {tool.description && (
                <Typography.Text
                  color="secondary"
                  size="sm"
                  css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}
                >
                  {tool.description}
                </Typography.Text>
              )}
            </button>

            {expandedIndex === index && (
              <div
                css={{
                  padding: `${theme.spacing.xs}px ${theme.spacing.md}px ${theme.spacing.md}px`,
                  paddingLeft: theme.spacing.md + theme.spacing.md + theme.spacing.sm,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                }}
              >
                {tool.annotations && Object.keys(tool.annotations).length > 0 && (
                  <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
                    {Object.entries(tool.annotations).map(([key, value]) => (
                      <Tag key={key} componentId="mlflow.mcp_registry.detail.tool_annotation_tag">
                        {key}: {String(value)}
                      </Tag>
                    ))}
                  </div>
                )}
                {tool.inputSchema && Object.keys(tool.inputSchema).length > 0 && (
                  <InputSchemaToggle data={tool.inputSchema} />
                )}
                {tool.outputSchema && Object.keys(tool.outputSchema).length > 0 && (
                  <OutputSchemaToggle data={tool.outputSchema} />
                )}
              </div>
            )}
          </div>
        ))}
        {hiddenCount > 0 && (
          <div
            css={{
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
              padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
              textAlign: 'center',
            }}
          >
            <Button
              componentId="mlflow.mcp_registry.detail.toggle_tools"
              type="link"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? (
                <FormattedMessage
                  defaultMessage="Show less"
                  description="MCP server version detail show less tools button"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Show {count} more"
                  description="MCP server version detail show more tools button"
                  values={{ count: hiddenCount }}
                />
              )}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

const useJSONToggle = (data: unknown) => {
  const [show, setShow] = useState(false);
  const jsonString = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return { show, setShow, jsonString };
};

const jsonPreStyles = (theme: ReturnType<typeof useDesignSystemTheme>['theme'], padding = theme.spacing.sm) =>
  ({
    margin: 0,
    padding,
    backgroundColor: theme.colors.backgroundSecondary,
    borderRadius: theme.borders.borderRadiusSm,
    overflow: 'auto' as const,
    fontSize: theme.typography.fontSizeSm,
  }) as const;

const InputSchemaToggle = ({ data }: { data: unknown }) => {
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

const OutputSchemaToggle = ({ data }: { data: unknown }) => {
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

const RawJSONToggle = ({ serverJson }: { serverJson: ServerJSONPayload }) => {
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
          <FormattedMessage
            defaultMessage="Hide raw server.json"
            description="MCP server version detail hide raw JSON button"
          />
        ) : (
          <FormattedMessage
            defaultMessage="View raw server.json"
            description="MCP server version detail view raw JSON button"
          />
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

const RawToolsJSONToggle = ({ tools }: { tools: MCPTool[] }) => {
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
          <FormattedMessage
            defaultMessage="Hide raw tools JSON"
            description="MCP server version detail hide raw tools JSON button"
          />
        ) : (
          <FormattedMessage
            defaultMessage="View raw tools JSON"
            description="MCP server version detail view raw tools JSON button"
          />
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
