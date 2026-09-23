import {
  BarsAscendingVerticalIcon,
  Button,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { TimelineTreeFilterButton } from './TimelineTreeFilterButton';
import type { SpanFilterState } from '../ModelTrace.types';

export const TimelineTreeHeader = ({
  showTimelineInfo,
  setShowTimelineInfo,
  spanFilterState,
  setSpanFilterState,
  showGraph,
  onToggleGraph,
}: {
  showTimelineInfo: boolean;
  setShowTimelineInfo: (showTimelineInfo: boolean) => void;
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
  showGraph?: boolean;
  onToggleGraph?: () => void;
}): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  return (
    <>
      <div
        css={{
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          paddingBottom: 3,
          boxSizing: 'border-box',
          minHeight: theme.spacing.xl + 2 * theme.spacing.sm,
          paddingLeft: theme.spacing.sm,
          alignItems: 'center',
          display: 'flex',
          justifyContent: 'space-between',
          gap: theme.spacing.xs,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            minWidth: 0,
          }}
        >
          <Typography.Text bold css={{ whiteSpace: 'nowrap' }}>
            <FormattedMessage
              defaultMessage="Spans"
              description="Header for the spans column within the MLflow trace UI"
            />
          </Typography.Text>
        </div>
        <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm, flexShrink: 0 }}>
          <Tooltip
            componentId="shared.model-trace-explorer.show-timeline-info-tooltip"
            content={
              showTimelineInfo ? (
                <FormattedMessage
                  defaultMessage="Hide execution timeline"
                  description="Tooltip for a button that hides execution timeline info in the trace UI."
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Show execution timeline"
                  description="Tooltip for a button that shows execution timeline info in the trace UI."
                />
              )
            }
          >
            <Button
              componentId="shared.model-trace-explorer.toggle-show-timeline"
              icon={<BarsAscendingVerticalIcon />}
              size="small"
              css={{ svg: { width: 14, height: 14 } }}
              type={showTimelineInfo ? 'primary' : undefined}
              aria-label={showTimelineInfo ? 'Hide execution timeline' : 'Show execution timeline'}
              aria-pressed={showTimelineInfo}
              onClick={() => setShowTimelineInfo(!showTimelineInfo)}
            />
          </Tooltip>
          <TimelineTreeFilterButton
            spanFilterState={spanFilterState}
            setSpanFilterState={setSpanFilterState}
            showGraph={showGraph}
            onToggleGraph={onToggleGraph}
          />
        </div>
      </div>
    </>
  );
};
