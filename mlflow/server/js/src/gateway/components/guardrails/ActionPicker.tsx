import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  NoIcon,
  SparkleIcon,
  Typography,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import type { GuardrailAction } from '../../types';

const ActionCard = ({
  componentId,
  value,
  icon,
  title,
  description,
  selected,
  onSelect,
}: {
  componentId: string;
  value: GuardrailAction;
  icon: React.ReactNode;
  title: React.ReactNode;
  description: React.ReactNode;
  selected: boolean;
  onSelect: (v: GuardrailAction) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId,
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  });

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      eventContext.onClick(e);
      onSelect(value);
    },
    [eventContext, onSelect, value],
  );

  return (
    <div
      role="option"
      aria-selected={selected}
      onClick={handleClick}
      onKeyDown={(e) => e.key === 'Enter' && onSelect(value)}
      tabIndex={0}
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        borderRadius: theme.borders.borderRadiusMd,
        border: `2px solid ${selected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
        cursor: 'pointer',
        '&:hover': { borderColor: theme.colors.actionPrimaryBackgroundDefault },
      }}
    >
      <div css={{ color: theme.colors.textSecondary, flexShrink: 0 }}>{icon}</div>
      <div>
        <Typography.Text bold css={{ display: 'block' }}>
          {title}
        </Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
          {description}
        </Typography.Text>
      </div>
    </div>
  );
};

export const ActionPicker = ({
  action,
  onActionChange,
}: {
  action: GuardrailAction;
  onActionChange: (a: GuardrailAction) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const actionOptions = [
    {
      value: 'VALIDATION' as const,
      componentId: 'mlflow.gateway.guardrails.action-option.validation',
      icon: <NoIcon />,
      title: <FormattedMessage defaultMessage="Block" description="Block action title" />,
      description: (
        <FormattedMessage
          defaultMessage="Return a 400 error with the guardrail name and reason. The original request or response is not passed through."
          description="Block action description"
        />
      ),
    },
    {
      value: 'SANITIZATION' as const,
      componentId: 'mlflow.gateway.guardrails.action-option.sanitization',
      icon: <SparkleIcon />,
      title: <FormattedMessage defaultMessage="Sanitize" description="Sanitize action title" />,
      description: (
        <FormattedMessage
          defaultMessage="Redact or mask flagged content, then allow the request to continue."
          description="Sanitize action description"
        />
      ),
    },
  ];
  return (
    <div>
      <Typography.Text
        bold
        css={{ display: 'block', fontSize: theme.typography.fontSizeLg, marginBottom: theme.spacing.sm }}
      >
        <FormattedMessage defaultMessage="Action" description="Guardrail action label" />
      </Typography.Text>
      <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
        {actionOptions.map(({ value, componentId, icon, title, description }) => (
          <ActionCard
            key={value}
            componentId={componentId}
            value={value}
            icon={icon}
            title={title}
            description={description}
            selected={action === value}
            onSelect={onActionChange}
          />
        ))}
      </div>
    </div>
  );
};
