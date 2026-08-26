import { type ReactNode, useState } from 'react';

import { Button, SparkleFillIcon, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';

export interface ModelTraceExplorerGenieButtonProps {
  componentId: string;
  onClick: () => void;
  disabled?: boolean;
  ariaLabel?: string;
  children?: ReactNode;
}

export const ModelTraceExplorerGenieButton = ({
  componentId,
  onClick,
  disabled,
  ariaLabel,
  children,
}: ModelTraceExplorerGenieButtonProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Button
      componentId={componentId}
      icon={
        <span
          css={{
            display: 'inline-flex',
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
      css={{
        '&&, &&:hover, &&:active': {
          borderColor: `${theme.colors.borderDecorative} !important`,
        },
      }}
    >
      {children}
    </Button>
  );
};
