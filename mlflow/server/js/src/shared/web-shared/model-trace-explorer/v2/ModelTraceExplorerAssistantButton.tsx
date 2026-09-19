import { type ReactNode, useState } from 'react';

import { Button, SparkleFillIcon, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';

import { getAiGradientBorderStyle } from '../../design-system/aiGradientBorderStyle';

export interface ModelTraceExplorerAssistantButtonProps {
  componentId: string;
  onClick: () => void;
  disabled?: boolean;
  ariaLabel?: string;
  children?: ReactNode;
}

export const ModelTraceExplorerAssistantButton = ({
  componentId,
  onClick,
  disabled,
  ariaLabel,
  children,
}: ModelTraceExplorerAssistantButtonProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Button
      componentId={componentId}
      icon={
        <span
          css={{
            display: 'inline-flex',
            marginRight: children ? theme.spacing.xs : 0,
            transition: 'transform 0.25s',
            transform: isHovered ? 'rotate(90deg)' : undefined,
          }}
        >
          {isHovered ? (
            <SparkleFillIcon css={{ svg: { width: 15, height: 15 } }} color="ai" />
          ) : (
            <SparkleIcon css={{ svg: { width: 15, height: 15 } }} color="ai" />
          )}
        </span>
      }
      disabled={disabled}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
      aria-label={ariaLabel}
      css={getAiGradientBorderStyle(theme)}
    >
      {children}
    </Button>
  );
};
