import { useMemo, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  SparkleFillIcon,
  SparkleIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useAssistant } from '../../../../assistant/AssistantContext';
import { useLogTelemetryEvent } from '../../../../telemetry/hooks/useLogTelemetryEvent';
import { COLLAPSED_CLASS_NAME, FULL_WIDTH_CLASS_NAME } from './constants';

export const ExperimentPageSideNavAssistantButton = () => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, isPanelOpen } = useAssistant();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => uuidv4(), []);
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={isPanelOpen}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        borderRadius: theme.borders.borderRadiusSm,
        cursor: 'pointer',
        marginBottom: theme.spacing.sm,
        backgroundColor: isPanelOpen ? theme.colors.actionDefaultBackgroundHover : undefined,
        color: isPanelOpen ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
        height: theme.typography.lineHeightBase,
        boxSizing: 'content-box',
        ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
      }}
      onClick={() => {
        openPanel();
        logTelemetryEvent({
          componentId: 'mlflow.experiment_side_nav.assistant_button',
          componentViewId: viewId,
          componentType: DesignSystemEventProviderComponentTypes.Button,
          componentSubType: null,
          eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
        });
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          openPanel();
        }
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Tooltip
        componentId="mlflow.experiment_side_nav.assistant_tooltip"
        content={<FormattedMessage defaultMessage="Assistant" description="Tooltip for assistant button" />}
        side="right"
        delayDuration={0}
      >
        <span
          className={COLLAPSED_CLASS_NAME}
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transform: isHovered ? 'rotate(-90deg)' : 'rotate(0deg)',
            transition: 'transform 0.3s ease',
          }}
        >
          {isHovered ? <SparkleFillIcon color="ai" /> : <SparkleIcon color="ai" />}
        </span>
      </Tooltip>
      <span
        className={FULL_WIDTH_CLASS_NAME}
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transform: isHovered ? 'rotate(-90deg)' : 'rotate(0deg)',
          transition: 'transform 0.3s ease',
        }}
      >
        {isHovered ? <SparkleFillIcon color="ai" /> : <SparkleIcon color="ai" />}
      </span>
      <Typography.Text className={FULL_WIDTH_CLASS_NAME} bold={isPanelOpen} color="primary">
        <FormattedMessage defaultMessage="Assistant" description="Sidebar button for AI assistant" />
      </Typography.Text>
      <Tag
        componentId="mlflow.experiment_side_nav.assistant_beta_tag"
        className={FULL_WIDTH_CLASS_NAME}
        color="turquoise"
        css={{ marginLeft: 'auto' }}
      >
        Beta
      </Tag>
    </div>
  );
};
