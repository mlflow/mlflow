import { v4 as uuidv4 } from 'uuid';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { WorkflowType } from '../contexts/WorkflowTypeContext';
import { FormattedMessage } from '@databricks/i18n';
import { useLogTelemetryEvent } from '../../telemetry/hooks/useLogTelemetryEvent';
import { useCallback, useMemo } from 'react';

export const MlflowSidebarWorkflowSwitch = ({
  workflowType,
  setWorkflowType,
}: {
  workflowType: WorkflowType;
  setWorkflowType: (workflowType: WorkflowType) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const viewId = useMemo(() => uuidv4(), []);

  const isGenAi = workflowType === WorkflowType.GENAI;
  const borderColor = isGenAi ? theme.gradients.aiBorderGradient : theme.colors.actionDefaultBorderDefault;

  const logTelemetryEvent = useLogTelemetryEvent();
  const handleChangeWorkflowType = useCallback(
    (workflowType: WorkflowType) => {
      setWorkflowType(workflowType);
      logTelemetryEvent({
        componentId: 'mlflow.sidebar.workflow_switch',
        componentViewId: viewId,
        componentType: DesignSystemEventProviderComponentTypes.ToggleButton,
        componentSubType: null,
        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
        value: workflowType,
      });
    },
    [setWorkflowType, logTelemetryEvent, viewId],
  );

  return (
    <Tooltip
      componentId="mlflow.sidebar.workflow_switch.tooltip"
      content={
        <FormattedMessage
          defaultMessage="Select your workflow type. This changes the tabs that are visible in the navigation sidebar."
          description="Tooltip for workflow switch"
        />
      }
      side="right"
    >
      <div
        css={{
          display: 'flex',
          width: 'min-content',
          position: 'relative',
          alignItems: 'center',
          background: theme.colors.backgroundSecondary,
          borderRadius: theme.borders.borderRadiusFull,
          border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        }}
      >
        {/* Sliding thumb indicator */}
        <div
          css={{
            position: 'absolute',
            top: -1,
            bottom: -1,
            left: -1,
            width: `${isGenAi ? 39 : 67}%`,
            background: borderColor,
            borderRadius: theme.borders.borderRadiusFull,
            padding: 1,
            transition: 'transform 200ms ease-out, width 200ms ease-out, background 200ms ease-out',
            transform: isGenAi ? 'translateX(0)' : `translateX(calc(51%))`,
            pointerEvents: 'none',
          }}
        >
          <div
            css={{
              width: '100%',
              height: '100%',
              background: theme.colors.backgroundPrimary,
              borderRadius: theme.borders.borderRadiusFull,
            }}
          />
        </div>

        {/* GenAI option */}
        <div
          css={{
            zIndex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            padding: theme.spacing.xs,
            paddingRight: theme.spacing.md,
            paddingLeft: theme.spacing.md,
            whiteSpace: 'nowrap' as const,
          }}
          role="button"
          tabIndex={0}
          onClick={() => handleChangeWorkflowType(WorkflowType.GENAI)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              handleChangeWorkflowType(WorkflowType.GENAI);
            }
          }}
        >
          <Typography.Text color={isGenAi ? undefined : 'secondary'}>
            <FormattedMessage defaultMessage="GenAI" description="Label for GenAI workflow type option" />
          </Typography.Text>
        </div>

        {/* Model training option */}
        <div
          css={{
            zIndex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            padding: theme.spacing.xs,
            paddingRight: theme.spacing.md,
            paddingLeft: theme.spacing.sm,
            whiteSpace: 'nowrap' as const,
          }}
          role="button"
          tabIndex={0}
          onClick={() => handleChangeWorkflowType(WorkflowType.MACHINE_LEARNING)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              handleChangeWorkflowType(WorkflowType.MACHINE_LEARNING);
            }
          }}
        >
          <Typography.Text color={isGenAi ? 'secondary' : undefined}>
            <FormattedMessage
              defaultMessage="Model training"
              description="Label for model training workflow type option"
            />
          </Typography.Text>
        </div>
      </div>
    </Tooltip>
  );
};
