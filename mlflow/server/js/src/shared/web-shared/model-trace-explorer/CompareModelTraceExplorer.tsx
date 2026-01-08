import { useCallback, useMemo, useState } from 'react';

import { Tabs, Tag, Typography, useDesignSystemTheme, Tooltip } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace } from './ModelTrace.types';
import {
  createTraceV4LongIdentifier,
  getModelTraceId,
  isV3ModelTraceInfo,
  doesTraceSupportV4API,
} from './ModelTraceExplorer.utils';
import { ModelTraceExplorerDetailCompareView } from './ModelTraceExplorerDetailCompareView';
import { ModelTraceExplorerGenericErrorState } from './ModelTraceExplorerGenericErrorState';
import { ModelTraceExplorerSummaryCompareView } from './summary-view/ModelTraceExplorerSummaryCompareView';
import { useCopyController } from '../copy';

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
}) => {
  const [activeView, setActiveView] = useState<'summary' | 'detail'>('summary');
  const { theme } = useDesignSystemTheme();

  const handleValueChange = useCallback((value: string) => {
    setActiveView(value as 'summary' | 'detail');
  }, []);

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
      <Tabs.Root
        componentId="shared.model-trace-explorer.view-mode-toggle"
        value={activeView}
        onValueChange={handleValueChange}
        css={{
          // this is to remove the margin at the bottom of the <Tabs> component
          '& > div:nth-of-type(1)': {
            marginBottom: 0,
            flexShrink: 0,
          },
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
        }}
      >
        <Tabs.List css={{ paddingLeft: theme.spacing.md, flexShrink: 0 }}>
          <Tabs.Trigger value="summary">
            <FormattedMessage
              defaultMessage="Summary"
              description="Label for the summary view tab in the model trace explorer"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="detail">
            <FormattedMessage
              defaultMessage="Details & Timeline"
              description="Label for the details & timeline view tab in the model trace explorer"
            />
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content
          value="summary"
          css={{
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
          }}
        >
          <ModelTraceExplorerSummaryCompareView modelTraces={modelTraces} />
        </Tabs.Content>
        <Tabs.Content
          value="detail"
          css={{
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
          }}
        >
          <ModelTraceExplorerDetailCompareView modelTraces={modelTraces} />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

export const CompareModelTraceExplorer = CompareModelTraceExplorerImpl;
