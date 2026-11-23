import { useMemo } from 'react';
import { SparkleDoubleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  estimatedCostMessages,
  formatCostUSD,
  formatEstimatedCostUSD,
} from '@databricks/web-shared/model-trace-explorer';
import { shouldEnableNewEvalRunsUi } from '../../common/utils/FeatureUtils';
import {
  OverviewChartContainer,
  OverviewChartEmptyState,
  OverviewChartErrorState,
  OverviewChartHeader,
  OverviewChartLoadingState,
} from '../../experiment-tracking/pages/experiment-overview/components/OverviewChartComponents';
import {
  SortableHeader,
  useSortState,
  useSummaryTableStyles,
} from '../../experiment-tracking/pages/experiment-overview/components/SummaryTableComponents';
import { formatCount, useChartColors } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';
import { OTHER_CALLERS, useGatewayFeatureUsageData } from '../hooks/useGatewayFeatureUsageData';
import { GATEWAY_FEATURES, GatewayFeatureLabels } from '../utils/gatewayFeatures';

type SortColumn = 'feature' | 'requests' | 'tokens' | 'cost';

// Matches the cost charts, which mark pricing-table costs as estimated.
const formatCost = (cost: number): string =>
  shouldEnableNewEvalRunsUi() ? formatEstimatedCostUSD(cost) : formatCostUSD(cost);

const FeatureNameCell = ({ name, color }: { name: string; color: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
      <div css={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: color, flexShrink: 0 }} />
      <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {name}
      </Typography.Text>
    </div>
  );
};

const RouteFeaturesHint = () => (
  <FormattedMessage
    defaultMessage="To track their usage here, use {modelUri} as the model for judges, the conversation simulator, issue detection or prompt optimization."
    description="Gateway usage > usage by feature > hint for routing MLflow features through an endpoint"
    values={{ modelUri: <code>gateway:/&lt;endpoint-name&gt;</code> }}
  />
);

/**
 * Gateway requests, tokens and cost for each built-in MLflow feature (LLM judges, the
 * conversation simulator, issue detection and prompt optimization), with every other caller
 * combined into an "Other" row.
 */
export const GatewayFeatureUsageSummary = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { getChartColor } = useChartColors();
  const { sortColumn, sortDirection, handleSort } = useSortState<SortColumn>('requests');
  const { headerRowStyle, bodyRowStyle, cellStyle } = useSummaryTableStyles('minmax(120px, 2fr) 1fr 1fr 1fr');
  const { usage, hasFeatureUsage, isLoading, error } = useGatewayFeatureUsageData();

  const rows = useMemo(() => {
    const labelOf = (feature: string) =>
      feature === OTHER_CALLERS
        ? intl.formatMessage({
            defaultMessage: 'Other',
            description: 'Gateway usage > usage by feature > row for requests not made by an MLflow feature',
          })
        : intl.formatMessage(GatewayFeatureLabels[feature as keyof typeof GatewayFeatureLabels]);

    const labeled = usage.map((row) => ({ ...row, label: labelOf(row.feature) }));
    const features = labeled.filter(({ feature }) => feature !== OTHER_CALLERS);
    features.sort((a, b) => {
      const comparison = sortColumn === 'feature' ? a.label.localeCompare(b.label) : a[sortColumn] - b[sortColumn];
      return sortDirection === 'asc' ? comparison : -comparison;
    });
    // "Other" is not a feature, so it stays last regardless of the sort.
    return [...features, ...labeled.filter(({ feature }) => feature === OTHER_CALLERS)];
  }, [usage, sortColumn, sortDirection, intl]);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  const sortableHeader = (column: SortColumn, label: React.ReactNode, centered = true) => (
    <SortableHeader
      column={column}
      sortColumn={sortColumn}
      sortDirection={sortDirection}
      onSort={handleSort}
      centered={centered}
    >
      {label}
    </SortableHeader>
  );

  return (
    <OverviewChartContainer componentId="mlflow.gateway.usage.feature_usage">
      <OverviewChartHeader
        icon={<SparkleDoubleIcon />}
        title={
          <FormattedMessage
            defaultMessage="Usage by Feature"
            description="Title for the gateway usage breakdown by built-in MLflow feature"
          />
        }
      />

      {rows.length > 0 ? (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          <div css={headerRowStyle}>
            {sortableHeader(
              'feature',
              <FormattedMessage defaultMessage="Feature" description="Column header for the MLflow feature name" />,
              false,
            )}
            {sortableHeader(
              'requests',
              <FormattedMessage defaultMessage="Requests" description="Column header for gateway request count" />,
            )}
            {sortableHeader(
              'tokens',
              <FormattedMessage defaultMessage="Tokens" description="Column header for gateway token count" />,
            )}
            {sortableHeader(
              'cost',
              <FormattedMessage defaultMessage="Cost" description="Column header for gateway cost" />,
            )}
          </div>
          {rows.map((row) => (
            <div key={row.feature} css={bodyRowStyle}>
              <FeatureNameCell
                name={row.label}
                color={
                  row.feature === OTHER_CALLERS
                    ? theme.colors.grey400
                    : getChartColor(GATEWAY_FEATURES.indexOf(row.feature))
                }
              />
              <Typography.Text css={cellStyle}>{formatCount(row.requests)}</Typography.Text>
              <Typography.Text css={cellStyle}>{formatCount(row.tokens)}</Typography.Text>
              <Typography.Text css={cellStyle}>{formatCost(row.cost)}</Typography.Text>
            </div>
          ))}
        </div>
      ) : (
        <OverviewChartEmptyState height={120} />
      )}

      {(!hasFeatureUsage || shouldEnableNewEvalRunsUi()) && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, marginTop: theme.spacing.sm }}>
          {!hasFeatureUsage && (
            <Typography.Text size="sm" color="secondary">
              <RouteFeaturesHint />
            </Typography.Text>
          )}
          {shouldEnableNewEvalRunsUi() && (
            <Typography.Text size="sm" color="secondary">
              <FormattedMessage {...estimatedCostMessages.tooltip} />
            </Typography.Text>
          )}
        </div>
      )}
    </OverviewChartContainer>
  );
};
