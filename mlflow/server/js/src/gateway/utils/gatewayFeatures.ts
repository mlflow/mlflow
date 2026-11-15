import { useCallback } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { defineMessage, type IntlShape, type MessageDescriptor } from 'react-intl';
import type { MetricFilterColumnOption } from '../../common/components/MetricsFilter.utils';
import { useChartColors } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';

/**
 * Built-in MLflow features that call the AI Gateway, identified by the caller they record on
 * each gateway trace. Mirrors `GatewayCaller` in mlflow/gateway/constants.py.
 */
export enum GatewayFeature {
  JUDGE = 'judge',
  SIMULATOR = 'simulator',
  ISSUE_DETECTION = 'issue_detection',
  PROMPT_OPTIMIZATION = 'prompt_optimization',
}

export const GATEWAY_FEATURES: GatewayFeature[] = Object.values(GatewayFeature);

export const isGatewayFeature = (caller: string): caller is GatewayFeature =>
  (GATEWAY_FEATURES as string[]).includes(caller);

export const GatewayFeatureLabels: Record<GatewayFeature, MessageDescriptor> = {
  [GatewayFeature.JUDGE]: defineMessage({
    defaultMessage: 'Judges',
    description: 'Gateway usage > MLflow feature name for LLM judges and scorers',
  }),
  [GatewayFeature.SIMULATOR]: defineMessage({
    defaultMessage: 'Conversation simulator',
    description: 'Gateway usage > MLflow feature name for the conversation simulator',
  }),
  [GatewayFeature.ISSUE_DETECTION]: defineMessage({
    defaultMessage: 'Issue detection',
    description: 'Gateway usage > MLflow feature name for issue detection',
  }),
  [GatewayFeature.PROMPT_OPTIMIZATION]: defineMessage({
    defaultMessage: 'Prompt optimization',
    description: 'Gateway usage > MLflow feature name for prompt optimization',
  }),
};

/**
 * Requests no built-in MLflow feature made, such as calls from applications or the playground, and
 * requests recorded without a caller.
 */
export const OTHER_CALLERS = 'other';

export type GatewayFeatureUsageKey = GatewayFeature | typeof OTHER_CALLERS;

export const GatewayFeatureUsageLabels: Record<GatewayFeatureUsageKey, MessageDescriptor> = {
  ...GatewayFeatureLabels,
  [OTHER_CALLERS]: defineMessage({
    defaultMessage: 'Other',
    description: 'Gateway usage > usage by feature > row for requests not made by an MLflow feature',
  }),
};

/** Color of each feature, and grey for "Other", shared by the usage chart and table. */
export const useGatewayFeatureUsageColor = () => {
  const { theme } = useDesignSystemTheme();
  const { getChartColor } = useChartColors();
  return useCallback(
    (key: GatewayFeatureUsageKey) =>
      key === OTHER_CALLERS ? theme.colors.grey400 : getChartColor(GATEWAY_FEATURES.indexOf(key)),
    [theme, getChartColor],
  );
};

/** "Feature" option for the gateway usage metrics filter, narrowing charts to one MLflow feature. */
export const getGatewayFeatureFilterOption = (intl: IntlShape): MetricFilterColumnOption => ({
  value: 'gateway_caller',
  label: intl.formatMessage({
    defaultMessage: 'Feature',
    description: 'Gateway usage > metrics filter > MLflow feature column option label',
  }),
  valueOptions: GATEWAY_FEATURES.map((feature) => ({
    value: feature,
    label: intl.formatMessage(GatewayFeatureLabels[feature]),
  })),
});
