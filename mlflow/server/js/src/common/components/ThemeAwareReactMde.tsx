import React from 'react';
import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import 'react-mde/lib/styles/css/react-mde-all.css';
import './ThemeAwareReactMde.css';
import ReactMde, { SvgIcon } from 'react-mde';

const getReactMdeIcon = (name: string) => <TooltipIcon name={name} />;

function TooltipIcon({ name }: { name: string }) {
  const { theme } = useDesignSystemTheme();
  return (
    <Tooltip side="top" content={name} componentId="mlflow.common.components.editable-note.tooltip-icon">
      <span css={{ color: theme.colors.textPrimary }}>
        <SvgIcon icon={name} />
      </span>
    </Tooltip>
  );
}

type ThemeAwareReactMdeProps = React.ComponentProps<typeof ReactMde>;

export function ThemeAwareReactMde(props: ThemeAwareReactMdeProps) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      className={theme.isDarkMode ? 'mlflow-dark-theme' : ''}
      css={{
        '--mlflow-dark-bg-primary': theme.colors.backgroundPrimary,
        '--mlflow-dark-bg-secondary': theme.colors.backgroundSecondary,
        '--mlflow-dark-bg-hover': theme.colors.actionDefaultBackgroundHover,
        '--mlflow-dark-text-primary': theme.colors.textPrimary,
        '--mlflow-dark-border': theme.colors.border,
      }}
    >
      <ReactMde getIcon={getReactMdeIcon} {...props} />
    </div>
  );
}
