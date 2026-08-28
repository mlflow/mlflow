import { CheckCircleIcon, Tag, Typography, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceGuardrailStatus } from '../ModelTrace.types';

export const ModelTraceExplorerGuardrailSpanView = ({ status }: { status: ModelTraceGuardrailStatus }) => {
  const { theme } = useDesignSystemTheme();
  const passed = status === 'passed';

  return (
    <div
      data-testid="model-trace-explorer-guardrail-span-view"
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        marginBottom: theme.spacing.sm,
        padding: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusSm,
      }}
    >
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Guardrail result"
          description="Label for the result of a guardrail span in the model trace explorer"
        />
      </Typography.Text>
      <Tag
        componentId="shared.model-trace-explorer.guardrail-status"
        icon={passed ? <CheckCircleIcon color="success" /> : <XCircleIcon color="danger" />}
      >
        {passed ? (
          <FormattedMessage
            defaultMessage="Passed"
            description="Status label for a guardrail span that allowed an operation"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Blocked"
            description="Status label for a guardrail span that blocked an operation"
          />
        )}
      </Tag>
    </div>
  );
};
