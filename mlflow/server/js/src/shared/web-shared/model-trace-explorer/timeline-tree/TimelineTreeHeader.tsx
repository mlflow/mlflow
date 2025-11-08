import {
  BarsAscendingVerticalIcon,
  ListBorderIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
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
}: {
  showTimelineInfo: boolean;
  setShowTimelineInfo: (showTimelineInfo: boolean) => void;
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        paddingBottom: 3,
        borderBottom: `1px solid ${theme.colors.border}`,
        boxSizing: 'border-box',
        paddingLeft: theme.spacing.sm,
        alignItems: 'center',
        display: 'flex',
        justifyContent: 'space-between',
      }}
    >
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="Trace breakdown"
          description="Header for the span tree within the MLflow trace UI"
        />
      </Typography.Text>
      <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm }}>
        <TimelineTreeFilterButton spanFilterState={spanFilterState} setSpanFilterState={setSpanFilterState} />
        <SegmentedControlGroup
          name="size-story"
          value={showTimelineInfo}
          onChange={(event) => {
            setShowTimelineInfo(event.target.value);
          }}
          size="small"
          componentId="shared.model-trace-explorer.toggle-show-timeline"
        >
          <SegmentedControlButton
            data-testid="hide-timeline-info-button"
            icon={
              <Tooltip
                componentId="shared.model-trace-explorer.hide-timeline-info-tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="Show span tree"
                    description="Tooltip for a button that show the span tree view of the trace UI."
                  />
                }
              >
                <ListBorderIcon />
              </Tooltip>
            }
            value={false}
          />
          <SegmentedControlButton
            data-testid="show-timeline-info-button"
            icon={
              <Tooltip
                componentId="shared.model-trace-explorer.show-timeline-info-tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="Show execution timeline"
                    description="Tooltip for a button that shows execution timeline info in the trace UI."
                  />
                }
              >
                <BarsAscendingVerticalIcon />
              </Tooltip>
            }
            value
          />
        </SegmentedControlGroup>
      </div>
    </div>
  );
};
