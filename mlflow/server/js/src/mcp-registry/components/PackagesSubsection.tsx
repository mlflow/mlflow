import { useState } from 'react';
import { resolveRunner } from '../installInstructions';
import { Button, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ConnectionSource } from '../types';
import type { ServerJSONPayload } from '../types';
import {
  borderedListContainerStyles,
  borderedListItemStyles,
  ellipsisStyles,
  flexRowStyles,
  showMoreRowStyles,
  noShrinkStyles,
  blockLabelStyles,
  monoFontStyles,
} from '../styles';
import { buildPackageConnectOptionKey } from '../utils';
import { ViewDetailsDrawer, DetailField, ArgumentList } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';
import { SubsectionHelpHeading } from './SubsectionHelpHeading';
import { ExpandableListSection } from './ExpandableListSection';
import { VisibilityToggle } from './VisibilityToggle';

type Package = NonNullable<ServerJSONPayload['packages']>[number];

const INITIAL_VISIBLE_PACKAGES = 5;

export const PackagesSubsection = ({
  packages,
  derivedName,
  showVisibilityControls,
  connectOptions,
  onToggleConnectOption,
}: {
  packages: Package[];
  derivedName: string;
  showVisibilityControls?: boolean;
  connectOptions?: Record<string, { hidden?: boolean }>;
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [showAll, setShowAll] = useState(false);
  const filteredPackages = showVisibilityControls
    ? packages
    : packages.filter((p) => !connectOptions?.[buildPackageConnectOptionKey(p)]?.hidden);
  const visiblePackages = showAll ? filteredPackages : filteredPackages.slice(0, INITIAL_VISIBLE_PACKAGES);
  const hiddenCount = filteredPackages.length - INITIAL_VISIBLE_PACKAGES;

  if (filteredPackages.length === 0) return null;

  return (
    <div>
      <SubsectionHelpHeading
        title={<FormattedMessage defaultMessage="Run locally" description="MCP server packages subsection heading" />}
        componentId="mlflow.mcp_registry.detail.packages_help"
        helpAriaLabel={intl.formatMessage({
          defaultMessage: 'About run locally',
          description: 'Aria label for packages subsection help popover',
        })}
        helpText={
          <FormattedMessage
            defaultMessage="Install and run this MCP server on your local machine using a package manager."
            description="Help text for MCP server packages subsection"
          />
        }
      />
      <ExpandableListSection
        items={visiblePackages}
        getKey={(pkg) => `${pkg.registryType}-${pkg.identifier}`}
        getAriaLabel={(pkg, expanded) =>
          intl.formatMessage(
            {
              defaultMessage: '{action} package {identifier}',
              description: 'Aria label for expanding/collapsing a package row',
            },
            { action: expanded ? 'Collapse' : 'Expand', identifier: pkg.identifier },
          )
        }
        getRowStyle={(pkg) =>
          connectOptions?.[buildPackageConnectOptionKey(pkg)]?.hidden ? { opacity: 0.5 } : undefined
        }
        renderRow={({ item: pkg }) => (
          <PackageRowContent
            pkg={pkg}
            showVisibilityControls={showVisibilityControls}
            isHidden={connectOptions?.[buildPackageConnectOptionKey(pkg)]?.hidden ?? false}
            onToggleVisibility={(visible) => onToggleConnectOption?.(buildPackageConnectOptionKey(pkg), visible)}
          />
        )}
        renderExpanded={(pkg) => <PackageExpandedContent pkg={pkg} derivedName={derivedName} />}
        footer={
          hiddenCount > 0 ? (
            <div css={showMoreRowStyles(theme)}>
              <Button
                componentId="mlflow.mcp_registry.detail.toggle_packages"
                type="link"
                onClick={() => setShowAll(!showAll)}
              >
                {showAll ? (
                  <FormattedMessage defaultMessage="Show less" description="Show less packages button" />
                ) : (
                  <FormattedMessage
                    defaultMessage="Show {count} more"
                    description="Show more packages button"
                    values={{ count: hiddenCount }}
                  />
                )}
              </Button>
            </div>
          ) : undefined
        }
      />
    </div>
  );
};

const PackageRowContent = ({
  pkg,
  showVisibilityControls,
  isHidden,
  onToggleVisibility,
}: {
  pkg: Package;
  showVisibilityControls?: boolean;
  isHidden: boolean;
  onToggleVisibility?: (visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVisible = !isHidden;
  const { runner } = resolveRunner(pkg.runtimeHint, pkg.registryType);

  return (
    <>
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
      {showVisibilityControls && (
        <VisibilityToggle
          componentId="mlflow.mcp_registry.detail.package"
          isVisible={isVisible}
          onToggle={(nowVisible) => onToggleVisibility?.(nowVisible)}
          showDisabledTag
          ariaLabel={intl.formatMessage(
            {
              defaultMessage: '{action} package {identifier}',
              description: 'Aria label for package visibility toggle',
            },
            { action: isVisible ? 'Hide' : 'Show', identifier: pkg.identifier },
          )}
        />
      )}
    </>
  );
};

const PackageExpandedContent = ({ pkg, derivedName }: { pkg: Package; derivedName: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const allEnvVars = pkg.environmentVariables ?? [];
  const transportLabel = pkg.transport?.type ?? 'stdio';

  return (
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
