import { defineMessage, type IntlShape, type MessageDescriptor } from 'react-intl';
import type { MetricFilterColumnOption } from '../../common/components/MetricsFilter.utils';

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
