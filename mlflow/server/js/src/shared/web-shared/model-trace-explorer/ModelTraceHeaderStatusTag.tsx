import React from 'react';
import { useIntl, FormattedMessage } from '@databricks/i18n';
import { ClockIcon, CheckCircleIcon, XCircleIcon, TagColors, useDesignSystemTheme } from '@databricks/design-system';

import { type ModelTraceState } from './ModelTrace.types';
import { ModelTraceHeaderMetricSection } from './ModelTraceExplorerMetricSection';

type Props = {
  statusState: ModelTraceState;
  getTruncatedLabel: (label: string) => string;
};

type StatusConfig = {
  label: string;
  icon: React.ReactNode;
  color: TagColors;
};

export const ModelTraceHeaderStatusTag = ({ statusState, getTruncatedLabel }: Props) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const getStatusConfig = (
    statusState: ModelTraceState,
    intl: ReturnType<typeof useIntl>,
    theme: ReturnType<typeof useDesignSystemTheme>['theme'],
  ): StatusConfig | null => {
    const statusMap: Record<ModelTraceState, StatusConfig | null> = {
      IN_PROGRESS: {
        label: intl.formatMessage({
          defaultMessage: 'In progress',
          description: 'Model trace header > status label > in progress',
        }),
        icon: <ClockIcon css={{ color: theme.colors.textValidationWarning }} />,
        color: 'lemon' as TagColors,
      },
      OK: {
        label: intl.formatMessage({ defaultMessage: 'OK', description: 'Model trace header > status label > ok' }),
        icon: <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />,
        color: 'teal' as TagColors,
      },
      ERROR: {
        label: intl.formatMessage({
          defaultMessage: 'Error',
          description: 'Model trace header > status label > error',
        }),
        icon: <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />,
        color: 'coral' as TagColors,
      },
      STATE_UNSPECIFIED: null,
    };
    return statusMap[statusState];
  };
  const status = getStatusConfig(statusState, intl, theme);

  if (!status) {
    return null;
  }

  return (
    <ModelTraceHeaderMetricSection
      label={<FormattedMessage defaultMessage="Status" description="Label for the status section" />}
      value={status.label}
      color={status.color}
      icon={status.icon}
      getTruncatedLabel={getTruncatedLabel}
      onCopy={() => {}}
    />
  );
};
