import { useMemo } from 'react';
import {
  Button,
  ExpandMoreIcon,
  Spacer,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { diffChars, diffJson, diffLines, diffWords } from 'diff';

import type { MCPAccessEndpoint, MCPIcon, MCPServerVersion, ServerJSONPayload } from '../types';
import { STATUS_TAG_COLOR } from '../utils';
import { MCPServerAliasesCell } from './MCPServerAliasesCell';
import { MCPServerIcon } from './MCPServerIcon';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import Utils from '../../common/utils/Utils';

const MetadataRow = ({ label, children }: { label: React.ReactNode; children: React.ReactNode }) => (
  <>
    <Typography.Text bold>{label}</Typography.Text>
    <div>{children}</div>
  </>
);

const VersionMetadataGrid = ({
  version,
  aliasesByVersion,
}: {
  version?: MCPServerVersion;
  aliasesByVersion: Record<string, string[]>;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (!version) return null;

  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: '120px 1fr',
        gridAutoRows: `minmax(${theme.typography.lineHeightLg}, auto)`,
        alignItems: 'flex-start',
        rowGap: theme.spacing.xs,
        columnGap: theme.spacing.sm,
      }}
    >
      <MetadataRow
        label={<FormattedMessage defaultMessage="Version:" description="MCP compare metadata version label" />}
      >
        <Typography.Text>{version.version}</Typography.Text>
      </MetadataRow>

      <MetadataRow
        label={<FormattedMessage defaultMessage="Status:" description="MCP compare metadata status label" />}
      >
        <Tag componentId="mlflow.mcp_registry.compare.status" color={STATUS_TAG_COLOR[version.status]}>
          {version.status}
        </Tag>
      </MetadataRow>

      <MetadataRow
        label={<FormattedMessage defaultMessage="Aliases:" description="MCP compare metadata aliases label" />}
      >
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
          {(aliasesByVersion[version.version] ?? []).length > 0 ? (
            <MCPServerAliasesCell aliases={aliasesByVersion[version.version] ?? []} />
          ) : (
            <Typography.Hint>—</Typography.Hint>
          )}
        </div>
      </MetadataRow>

      <MetadataRow
        label={<FormattedMessage defaultMessage="Created:" description="MCP compare metadata created label" />}
      >
        <Typography.Text>
          {version.creation_timestamp ? Utils.formatTimestamp(version.creation_timestamp, intl) : '—'}
        </Typography.Text>
      </MetadataRow>

      {version.created_by && (
        <MetadataRow
          label={<FormattedMessage defaultMessage="Created by:" description="MCP compare metadata created by label" />}
        >
          <Typography.Text>{version.created_by}</Typography.Text>
        </MetadataRow>
      )}

      {version.last_updated_timestamp && (
        <MetadataRow
          label={<FormattedMessage defaultMessage="Updated:" description="MCP compare metadata updated label" />}
        >
          <Typography.Text>{Utils.formatTimestamp(version.last_updated_timestamp, intl)}</Typography.Text>
        </MetadataRow>
      )}

      {version.last_updated_by && (
        <MetadataRow
          label={<FormattedMessage defaultMessage="Updated by:" description="MCP compare metadata updated by label" />}
        >
          <Typography.Text>{version.last_updated_by}</Typography.Text>
        </MetadataRow>
      )}

      {version.server_json?.['icons'] && (version.server_json['icons'] as MCPIcon[]).length > 0 && (
        <MetadataRow label={<FormattedMessage defaultMessage="Icon:" description="MCP compare metadata icon label" />}>
          <MCPServerIcon icons={version.server_json['icons'] as MCPIcon[]} name={version.server_json.name} />
        </MetadataRow>
      )}

      <MetadataRow
        label={<FormattedMessage defaultMessage="Metadata:" description="MCP compare metadata tags label" />}
      >
        {Object.keys(version.tags ?? {}).length > 0 ? (
          <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
            {Object.entries(version.tags ?? {}).map(([key, value]) => (
              <KeyValueTag css={{ margin: 0 }} key={key} tag={{ key, value }} />
            ))}
          </div>
        ) : (
          <Typography.Hint>—</Typography.Hint>
        )}
      </MetadataRow>
    </div>
  );
};

const EXTRACTED_FIELDS = ['description', 'packages', 'remotes', 'repository', 'icons', 'title', 'websiteUrl'] as const;

const SKIPPED_FIELDS = ['name', '$schema', '_meta', 'version'] as const;

const extractServerJsonSections = (serverJson?: ServerJSONPayload) => {
  if (!serverJson) return { fields: {} as Record<string, unknown>, extra: undefined };
  const fields: Record<string, unknown> = {};
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(serverJson)) {
    if ((EXTRACTED_FIELDS as readonly string[]).includes(key) && value != null) {
      fields[key] = value;
    } else if (!(SKIPPED_FIELDS as readonly string[]).includes(key)) {
      extra[key] = value;
    }
  }
  return { fields, extra: Object.keys(extra).length > 0 ? extra : undefined };
};

const stringify = (value: unknown): string => {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
};

const stringifyEndpoints = (endpoints?: MCPAccessEndpoint[]): string => {
  if (!endpoints?.length) return '';
  return endpoints
    .map(
      (b) =>
        `${b.transport_type} ${b.url}${b.server_alias ? ` (alias: ${b.server_alias})` : ''}${b.server_version ? ` (version: ${b.server_version})` : ''}`,
    )
    .join('\n');
};

type DiffMode = 'words' | 'chars' | 'lines' | 'json';

interface DiffEntry {
  key: string;
  label: React.ReactNode;
  baselineText: string;
  comparedText: string;
  showWhenEmpty?: boolean;
  diffMode?: DiffMode;
}

const computeTextDiff = (baseline: string, compared: string, mode: DiffMode) => {
  switch (mode) {
    case 'chars':
      return diffChars(baseline, compared);
    case 'lines':
      return diffLines(baseline, compared);
    case 'json':
      try {
        return diffJson(JSON.parse(baseline || '{}'), JSON.parse(compared || '{}'));
      } catch {
        return diffWords(baseline, compared);
      }
    default:
      return diffWords(baseline, compared);
  }
};

const TextDiffPanel = ({
  baselineText,
  comparedText,
  emptyFallback,
  colors,
  diffMode = 'words',
}: {
  baselineText: string;
  comparedText: string;
  emptyFallback: string;
  colors: { addedBackground: string; removedBackground: string };
  diffMode?: DiffMode;
}) => {
  const { theme } = useDesignSystemTheme();
  const diff = useMemo(
    () => computeTextDiff(baselineText, comparedText, diffMode) ?? [],
    [baselineText, comparedText, diffMode],
  );

  const preStyles = {
    flex: 1,
    margin: 0,
    padding: theme.spacing.md,
    backgroundColor: theme.colors.backgroundSecondary,
    borderRadius: theme.borders.borderRadiusSm,
    overflow: 'auto' as const,
    fontSize: theme.typography.fontSizeSm,
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
  };

  return (
    <div css={{ display: 'flex', alignItems: 'stretch' }}>
      <pre css={preStyles}>
        <code>{baselineText || emptyFallback}</code>
      </pre>
      <div css={{ width: theme.spacing.sm }} />
      <pre css={preStyles}>
        <code>
          {diff.map((part, index) => (
            <span
              key={index}
              css={{
                backgroundColor: part.added
                  ? colors.addedBackground
                  : part.removed
                    ? colors.removedBackground
                    : undefined,
                textDecoration: part.removed ? 'line-through' : 'none',
              }}
            >
              {part.value}
            </span>
          ))}
          {!comparedText && emptyFallback}
        </code>
      </pre>
    </div>
  );
};

const DIFF_CONFIGS: { key: string; label: React.ReactNode; diffMode?: DiffMode }[] = [
  {
    key: '_displayName',
    diffMode: 'words',
    label: <FormattedMessage defaultMessage="Display name" description="MCP compare display name heading" />,
  },
  {
    key: '_source',
    diffMode: 'chars',
    label: <FormattedMessage defaultMessage="Source" description="MCP compare source heading" />,
  },
  {
    key: '_endpoints',
    diffMode: 'lines',
    label: <FormattedMessage defaultMessage="Access endpoints" description="MCP compare endpoints heading" />,
  },
  {
    key: 'title',
    diffMode: 'words',
    label: <FormattedMessage defaultMessage="Title" description="MCP compare title heading" />,
  },
  {
    key: 'description',
    diffMode: 'words',
    label: <FormattedMessage defaultMessage="Description" description="MCP compare description heading" />,
  },
  {
    key: 'websiteUrl',
    diffMode: 'chars',
    label: <FormattedMessage defaultMessage="Website URL" description="MCP compare websiteUrl heading" />,
  },
  {
    key: 'repository',
    diffMode: 'json',
    label: <FormattedMessage defaultMessage="Repository" description="MCP compare repository heading" />,
  },
  {
    key: 'remotes',
    diffMode: 'json',
    label: <FormattedMessage defaultMessage="Official endpoints" description="MCP compare remotes heading" />,
  },
  {
    key: 'packages',
    diffMode: 'json',
    label: <FormattedMessage defaultMessage="Local packages" description="MCP compare packages heading" />,
  },
  {
    key: '_extra',
    diffMode: 'json',
    label: <FormattedMessage defaultMessage="Configuration" description="MCP compare extra config heading" />,
  },
  {
    key: '_tools',
    diffMode: 'json',
    label: <FormattedMessage defaultMessage="Tools" description="MCP compare tools heading" />,
  },
];

export const MCPServerVersionCompare = ({
  baselineVersion,
  comparedVersion,
  aliasesByVersion,
  baselineEndpoints,
  comparedEndpoints,
  onSwitchSides,
}: {
  baselineVersion?: MCPServerVersion;
  comparedVersion?: MCPServerVersion;
  aliasesByVersion: Record<string, string[]>;
  baselineEndpoints?: MCPAccessEndpoint[];
  comparedEndpoints?: MCPAccessEndpoint[];
  onSwitchSides: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const baselineSplit = useMemo(
    () => extractServerJsonSections(baselineVersion?.server_json),
    [baselineVersion?.server_json],
  );
  const comparedSplit = useMemo(
    () => extractServerJsonSections(comparedVersion?.server_json),
    [comparedVersion?.server_json],
  );

  const allDiffs = useMemo<DiffEntry[]>(() => {
    const getText = (key: string, side: 'baseline' | 'compared'): string => {
      const split = side === 'baseline' ? baselineSplit : comparedSplit;
      const version = side === 'baseline' ? baselineVersion : comparedVersion;
      const endpoints = side === 'baseline' ? baselineEndpoints : comparedEndpoints;
      switch (key) {
        case '_displayName':
          return version?.display_name ?? '';
        case '_source':
          return version?.source ?? '';
        case '_endpoints':
          return stringifyEndpoints(endpoints);
        case '_tools':
          return version?.tools?.length ? stringify(version.tools) : '';
        case '_extra':
          return stringify(split.extra);
        default:
          return stringify(split.fields[key]);
      }
    };

    return DIFF_CONFIGS.map(({ key, label, diffMode }) => ({
      key,
      label,
      baselineText: getText(key, 'baseline'),
      comparedText: getText(key, 'compared'),
      showWhenEmpty: key === '_tools',
      diffMode,
    }));
  }, [baselineSplit, comparedSplit, baselineVersion, comparedVersion, baselineEndpoints, comparedEndpoints]);

  const { changed, identical } = useMemo(() => {
    const changedEntries: DiffEntry[] = [];
    const identicalLabels: React.ReactNode[] = [];

    for (const entry of allDiffs) {
      const hasBaseline = Boolean(entry.baselineText);
      const hasCompared = Boolean(entry.comparedText);
      if (!hasBaseline && !hasCompared && !entry.showWhenEmpty) continue;

      if (entry.baselineText === entry.comparedText && !entry.showWhenEmpty) {
        if (hasBaseline) identicalLabels.push(entry.label);
      } else {
        changedEntries.push(entry);
      }
    }
    return { changed: changedEntries, identical: identicalLabels };
  }, [allDiffs]);

  const colors = useMemo(
    () => ({
      addedBackground: theme.isDarkMode ? theme.colors.green700 : theme.colors.green300,
      removedBackground: theme.isDarkMode ? theme.colors.red700 : theme.colors.red300,
    }),
    [theme],
  );

  const emptyFallback = intl.formatMessage({
    defaultMessage: 'Empty',
    description: 'Fallback for empty content in compare view',
  });

  const switchButton = (
    <Tooltip
      componentId="mlflow.mcp_registry.compare.switch_sides.tooltip"
      content={
        <FormattedMessage
          defaultMessage="Switch sides"
          description="Label for button to switch MCP server versions in comparison view"
        />
      }
      side="top"
    >
      <Button
        aria-label={intl.formatMessage({
          defaultMessage: 'Switch sides',
          description: 'Label for button to switch MCP server versions in comparison view',
        })}
        componentId="mlflow.mcp_registry.compare.switch_sides"
        icon={<ExpandMoreIcon css={{ svg: { rotate: '90deg' } }} />}
        onClick={onSwitchSides}
      />
    </Tooltip>
  );

  return (
    <div
      css={{
        flex: 1,
        padding: theme.spacing.md,
        paddingTop: 0,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      <Typography.Title level={3}>
        <FormattedMessage
          defaultMessage="Comparing version {baseline} with version {compared}"
          description="MCP server version compare heading"
          values={{
            baseline: baselineVersion?.version,
            compared: comparedVersion?.version,
          }}
        />
      </Typography.Title>

      <div css={{ display: 'flex' }}>
        <div css={{ flex: 1 }}>
          <VersionMetadataGrid version={baselineVersion} aliasesByVersion={aliasesByVersion} />
        </div>
        <div css={{ paddingInline: theme.spacing.sm, display: 'flex', alignItems: 'flex-start' }}>{switchButton}</div>
        <div css={{ flex: 1 }}>
          <VersionMetadataGrid version={comparedVersion} aliasesByVersion={aliasesByVersion} />
        </div>
      </div>

      {identical.length > 0 && (
        <div>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Identical:" description="MCP compare identical sections heading" />
          </Typography.Text>
          <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
            {identical.map((label, i) => (
              <span key={i}>
                {i > 0 && ', '}
                {label}
              </span>
            ))}
          </Typography.Text>
        </div>
      )}

      {changed.map(({ key, label, baselineText, comparedText, diffMode }) => (
        <div key={key}>
          <Typography.Text bold>{label}</Typography.Text>
          <Spacer shrinks={false} size="sm" />
          <TextDiffPanel
            baselineText={baselineText}
            comparedText={comparedText}
            emptyFallback={emptyFallback}
            colors={colors}
            diffMode={diffMode}
          />
        </div>
      ))}
    </div>
  );
};
