import { useState } from 'react';
import { resolveRunner } from '../installInstructions';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  InfoIcon,
  Popover,
  Tag,
  Tooltip,
  Typography,
  VisibleIcon,
  VisibleOffIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ConnectionSource } from '../types';
import type { ServerJSONPayload } from '../types';
import {
  expandableRowButtonStyles,
  chevronContainerStyles,
  borderedSectionContainerStyles,
  borderedListContainerStyles,
  borderedListItemStyles,
  expandedContentPanelStyles,
  popoverTriggerStyles,
  ellipsisStyles,
  sectionHeadingRowStyles,
  flexRowStyles,
  showMoreRowStyles,
  noShrinkStyles,
  blockLabelStyles,
  monoFontStyles,
} from '../styles';
import { buildPackageConnectOptionKey } from '../utils';
import { ViewDetailsDrawer, DetailField, ArgumentList } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';

const INITIAL_VISIBLE_PACKAGES = 5;

export const PackagesSubsection = ({
  packages,
  derivedName,
  isAdmin,
  isAuthAvailable,
  connectOptions,
  onToggleConnectOption,
}: {
  packages: NonNullable<ServerJSONPayload['packages']>;
  derivedName: string;
  isAdmin?: boolean;
  isAuthAvailable?: boolean;
  connectOptions?: Record<string, { hidden?: boolean }>;
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const showVisibilityControls = isAuthAvailable && isAdmin;
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const visiblePackages = showAll ? packages : packages.slice(0, INITIAL_VISIBLE_PACKAGES);
  const hiddenCount = packages.length - INITIAL_VISIBLE_PACKAGES;

  return (
    <div>
      <div css={sectionHeadingRowStyles(theme)}>
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Run locally" description="MCP server packages subsection heading" />
        </Typography.Text>
        <Popover.Root componentId="mlflow.mcp_registry.detail.packages_help">
          <Popover.Trigger
            css={popoverTriggerStyles(theme)}
            aria-label={intl.formatMessage({
              defaultMessage: 'About run locally',
              description: 'Aria label for packages subsection help popover',
            })}
          >
            <InfoIcon />
          </Popover.Trigger>
          <Popover.Content align="start" css={{ maxWidth: 360 }}>
            <Typography.Paragraph withoutMargins>
              <FormattedMessage
                defaultMessage="Install and run this MCP server on your local machine using a package manager."
                description="Help text for MCP server packages subsection"
              />
            </Typography.Paragraph>
            <Popover.Arrow />
          </Popover.Content>
        </Popover.Root>
      </div>
      <div css={borderedSectionContainerStyles(theme)}>
        {visiblePackages.map((pkg, index) => (
          <PackageRow
            key={`${pkg.registryType}-${pkg.identifier}`}
            pkg={pkg}
            derivedName={derivedName}
            expanded={expandedIndex === index}
            onToggle={() => setExpandedIndex(expandedIndex === index ? null : index)}
            showTopBorder={index > 0}
            showVisibilityControls={showVisibilityControls}
            isHidden={connectOptions?.[buildPackageConnectOptionKey(pkg)]?.hidden ?? false}
            onToggleVisibility={(visible) => onToggleConnectOption?.(buildPackageConnectOptionKey(pkg), visible)}
          />
        ))}
        {hiddenCount > 0 && (
          <div css={showMoreRowStyles(theme)}>
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
  derivedName,
  expanded,
  onToggle,
  showTopBorder,
  showVisibilityControls,
  isHidden,
  onToggleVisibility,
}: {
  pkg: NonNullable<ServerJSONPayload['packages']>[number];
  derivedName: string;
  expanded: boolean;
  onToggle: () => void;
  showTopBorder: boolean;
  showVisibilityControls?: boolean;
  isHidden?: boolean;
  onToggleVisibility?: (visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVisible = !isHidden;
  const allEnvVars = pkg.environmentVariables ?? [];
  const transportLabel = pkg.transport?.type ?? 'stdio';
  const { runner } = resolveRunner(pkg.runtimeHint, pkg.registryType);
  const isDisabled = !isVisible;

  if (isDisabled && !showVisibilityControls) return null;

  return (
    <div
      css={{
        borderTop: showTopBorder ? `1px solid ${theme.colors.border}` : 'none',
        opacity: isDisabled ? 0.5 : 1,
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
        css={expandableRowButtonStyles(theme)}
      >
        <div css={chevronContainerStyles(theme)}>{expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}</div>
        <Tag componentId="mlflow.mcp_registry.detail.package_registry_tag" color="turquoise" css={noShrinkStyles}>
          {pkg.registryType}
        </Tag>
        <Typography.Text color="secondary" css={ellipsisStyles(theme)}>
          <FormattedMessage
            defaultMessage="<strong>Run locally with {runner}</strong> {identifier}"
            description="Package row description showing runner and identifier"
            values={{
              runner: runner ?? pkg.registryType,
              identifier: pkg.identifier,
              strong: (text: string) => <strong>{text}</strong>,
            }}
          />
        </Typography.Text>
        {showVisibilityControls && isDisabled && (
          <Tag componentId="mlflow.mcp_registry.detail.package.disabled_tag" color="charcoal" css={noShrinkStyles}>
            <FormattedMessage defaultMessage="Disabled" description="Label for disabled package" />
          </Tag>
        )}
        {showVisibilityControls && (
          <div css={noShrinkStyles} onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
            <Tooltip
              componentId="mlflow.mcp_registry.detail.package.visibility_tooltip"
              content={
                isVisible ? (
                  <FormattedMessage
                    defaultMessage="Visible to developers. Click to hide."
                    description="Tooltip for visible package toggle"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Hidden from developers. Click to show."
                    description="Tooltip for hidden package toggle"
                  />
                )
              }
            >
              <Button
                componentId="mlflow.mcp_registry.detail.package.visibility_row"
                type="tertiary"
                size="small"
                icon={isVisible ? <VisibleIcon /> : <VisibleOffIcon />}
                onClick={() => onToggleVisibility?.(!!isHidden)}
                aria-label={intl.formatMessage(
                  {
                    defaultMessage: '{action} package {identifier}',
                    description: 'Aria label for package visibility toggle',
                  },
                  { action: isVisible ? 'Hide' : 'Show', identifier: pkg.identifier },
                )}
              />
            </Tooltip>
          </div>
        )}
      </button>

      {expanded && (
        <div css={expandedContentPanelStyles(theme)}>
          <ConnectionInstructions
            source={ConnectionSource.PACKAGE}
            pkg={pkg}
            derivedName={derivedName}
            detailLink={
              <ViewDetailsDrawer title={pkg.identifier}>
                <DetailField
                  label={intl.formatMessage({ defaultMessage: 'Identifier', description: 'Package identifier label' })}
                  value={pkg.identifier}
                  mono
                />
                <DetailField
                  label={intl.formatMessage({
                    defaultMessage: 'Registry type',
                    description: 'Package registry type label',
                  })}
                  value={pkg.registryType}
                  tagColor="turquoise"
                />
                {pkg.version && (
                  <DetailField
                    label={intl.formatMessage({ defaultMessage: 'Version', description: 'Package version label' })}
                    value={pkg.version}
                    mono
                  />
                )}
                <DetailField
                  label={intl.formatMessage({ defaultMessage: 'Transport', description: 'Package transport label' })}
                  value={transportLabel}
                  tagColor="default"
                />
                {pkg.runtimeHint && (
                  <DetailField
                    label={intl.formatMessage({
                      defaultMessage: 'Runtime hint',
                      description: 'Package runtime hint label',
                    })}
                    value={pkg.runtimeHint}
                    mono
                  />
                )}
                {pkg.registryBaseUrl && (
                  <DetailField
                    label={intl.formatMessage({
                      defaultMessage: 'Registry URL',
                      description: 'Package registry URL label',
                    })}
                    value={pkg.registryBaseUrl}
                    mono
                    link
                  />
                )}
                {pkg.fileSha256 && (
                  <DetailField
                    label={intl.formatMessage({ defaultMessage: 'SHA-256', description: 'Package file hash label' })}
                    value={pkg.fileSha256}
                    mono
                  />
                )}
                {pkg.transport?.url && (
                  <DetailField
                    label={intl.formatMessage({
                      defaultMessage: 'Transport URL',
                      description: 'Package transport URL label',
                    })}
                    value={pkg.transport.url}
                    mono
                    link
                  />
                )}
                {allEnvVars.length > 0 && <EnvVarList envVars={allEnvVars} />}
                {pkg.runtimeArguments && pkg.runtimeArguments.length > 0 && (
                  <ArgumentList
                    label={intl.formatMessage({
                      defaultMessage: 'Runtime arguments',
                      description: 'Package runtime arguments label',
                    })}
                    args={pkg.runtimeArguments}
                  />
                )}
                {pkg.packageArguments && pkg.packageArguments.length > 0 && (
                  <ArgumentList
                    label={intl.formatMessage({
                      defaultMessage: 'Package arguments',
                      description: 'Package arguments label',
                    })}
                    args={pkg.packageArguments}
                  />
                )}
              </ViewDetailsDrawer>
            }
          />
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
      <Typography.Text bold size="sm" css={blockLabelStyles(theme)}>
        <FormattedMessage
          defaultMessage="Environment Variables ({count})"
          description="MCP server package environment variables heading with count"
          values={{ count: vars.length }}
        />
      </Typography.Text>
      <div css={borderedListContainerStyles(theme)}>
        {visibleVars.map((envVar, i) => (
          <div key={envVar.name} css={borderedListItemStyles(theme, i > 0)}>
            <div css={flexRowStyles(theme)}>
              <Typography.Text bold size="sm" css={monoFontStyles}>
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
          <div css={showMoreRowStyles(theme)}>
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
