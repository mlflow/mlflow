import { useMemo } from 'react';

import { Tag, Typography, useDesignSystemTheme, Tooltip } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace } from './ModelTrace.types';
import { createTraceV4LongIdentifier, getModelTraceId, isV3ModelTraceInfo } from './ModelTraceExplorer.utils';
import { doesTraceSupportV4API } from '../../genai-traces-table/utils/TraceLocationUtils';
import { ModelTraceExplorerDetailCompareView } from '../ModelTraceExplorerDetailCompareView';
import { ModelTraceExplorerGenericErrorState } from '../ModelTraceExplorerGenericErrorState';
import { useCopyController } from '../../copy/useCopyController';

const IdentifierTag = ({ modelTrace }: { modelTrace: ModelTrace }) => {
  const [modelTraceId, modelTraceIdToDisplay] = useMemo(() => {
    const modelTraceInfo = modelTrace.info;
    if (doesTraceSupportV4API(modelTraceInfo) && isV3ModelTraceInfo(modelTraceInfo)) {
      return [createTraceV4LongIdentifier(modelTraceInfo), modelTraceInfo.trace_id];
    }
    return [isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : (modelTraceInfo.request_id ?? '')];
  }, [modelTrace]);

  const { copy, tooltipMessage } = useCopyController(modelTraceId, modelTraceId);
  const { theme } = useDesignSystemTheme();

  return (
    <Tooltip content={tooltipMessage} componentId="shared.model-trace-explorer.compare-modal.trace-id-tag-tooltip">
      <Tag
        componentId="shared.model-trace-explorer.compare-modal.trace-id-tag"
        onClick={copy}
        css={{ marginRight: 0 }}
        color="purple"
      >
        <span css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: theme.spacing.xs }}>
          <span>{modelTraceIdToDisplay ?? modelTraceId}</span>
        </span>
      </Tag>
    </Tooltip>
  );
};

export const CompareModelTraceExplorerImpl = ({
  modelTraces,
  className,
}: {
  modelTraces: ModelTrace[];
  className?: string;
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ overflow: 'hidden', height: '100%', display: 'flex', flexDirection: 'column' }} className={className}>
      <div
        css={{
          paddingLeft: theme.spacing.md,
          paddingBottom: theme.spacing.sm,
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.sm,
          alignItems: 'center',
        }}
      >
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Compared IDs:"
            description="Label for the compared IDs section in the model trace explorer"
          />
        </Typography.Hint>
        {modelTraces.map((modelTrace) => (
          <IdentifierTag key={getModelTraceId(modelTrace)} modelTrace={modelTrace} />
        ))}
      </div>
      <ModelTraceExplorerDetailCompareView modelTraces={modelTraces} />
    </div>
  );
};

export const CompareModelTraceExplorer: typeof CompareModelTraceExplorerImpl = CompareModelTraceExplorerImpl;
