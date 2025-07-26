import React, { Component, ErrorInfo, ReactNode } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, useDesignSystemTheme } from '@databricks/design-system';

interface Props {
  children: ReactNode;
  fallbackComponent?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * Error boundary for RunNameHeaderCellRenderer to gracefully handle rendering errors
 * without breaking the entire runs table
 */
export class RunNameHeaderCellRendererErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('RunNameHeaderCellRenderer error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: undefined });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallbackComponent) {
        return this.props.fallbackComponent;
      }

      return <RunNameHeaderErrorFallback onReset={this.handleReset} error={this.state.error} />;
    }

    return this.props.children;
  }
}

interface FallbackProps {
  onReset: () => void;
  error?: Error;
}

const RunNameHeaderErrorFallback: React.FC<FallbackProps> = ({ onReset, error }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        padding: theme.spacing.xs,
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
      }}
    >
      <span>
        <FormattedMessage
          defaultMessage="Run Name"
          description="Fallback text when RunNameHeaderCellRenderer fails to render"
        />
      </span>
      <Button
        componentId="run-name-header-error-retry"
        type="tertiary"
        size="small"
        onClick={onReset}
        css={{
          fontSize: theme.typography.fontSizeSm,
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        }}
      >
        <FormattedMessage defaultMessage="Retry" description="Button to retry loading the runs visibility dropdown" />
      </Button>
    </div>
  );
};
