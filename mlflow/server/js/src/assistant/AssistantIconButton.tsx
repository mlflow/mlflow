import { useMemo, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  SparkleFillIcon,
  SparkleIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useAssistant } from './AssistantContext';
import { useLogTelemetryEvent } from '../telemetry/hooks/useLogTelemetryEvent';

/**
 * Animated sparkle icon that rotates and fills on hover.
 * Use this when you need the icon as part of a larger clickable area.
 */
interface AssistantSparkleIconProps {
  isHovered: boolean;
  iconSize?: number;
  className?: string;
}

export const AssistantSparkleIcon = ({ isHovered, iconSize, className }: AssistantSparkleIconProps) => {
  const iconCss = iconSize ? { fontSize: iconSize } : undefined;

  return (
    <span
      className={className}
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transform: isHovered ? 'rotate(-90deg)' : 'rotate(0deg)',
        transition: 'transform 0.3s ease',
      }}
    >
      {isHovered ? <SparkleFillIcon color="ai" css={iconCss} /> : <SparkleIcon color="ai" css={iconCss} />}
    </span>
  );
};

/**
 * Standalone icon button for the assistant.
 * Use this for icon-only buttons (e.g., in the header).
 */
interface AssistantIconButtonProps {
  componentId: string;
  tooltipSide?: 'top' | 'bottom' | 'left' | 'right';
  iconSize?: number;
  className?: string;
}

export const AssistantIconButton = ({
  componentId,
  tooltipSide = 'bottom',
  iconSize,
  className,
}: AssistantIconButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, closePanel, isPanelOpen } = useAssistant();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => uuidv4(), []);
  const [isHovered, setIsHovered] = useState(false);

  const togglePanel = () => {
    if (isPanelOpen) {
      closePanel();
    } else {
      openPanel();
    }
    logTelemetryEvent({
      componentId,
      componentViewId: viewId,
      componentType: DesignSystemEventProviderComponentTypes.Button,
      componentSubType: null,
      eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      togglePanel();
    }
  };

  return (
    <Tooltip
      componentId={`${componentId}.tooltip`}
      content={<FormattedMessage defaultMessage="Assistant" description="Tooltip for assistant button" />}
      side={tooltipSide}
      delayDuration={0}
    >
      <div
        role="button"
        tabIndex={0}
        aria-pressed={isPanelOpen}
        className={className}
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.xs,
          borderRadius: theme.borders.borderRadiusSm,
          cursor: 'pointer',
          backgroundColor: isPanelOpen ? theme.colors.actionDefaultBackgroundHover : undefined,
          color: isPanelOpen ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
          ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
        }}
        onClick={togglePanel}
        onKeyDown={handleKeyDown}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <AssistantSparkleIcon isHovered={isHovered} iconSize={iconSize} />
      </div>
    </Tooltip>
  );
};
