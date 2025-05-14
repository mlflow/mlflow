import React, { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, type ButtonProps, LegacyTooltip } from '@databricks/design-system';

interface CopyButtonProps extends Partial<ButtonProps> {
  copyText: string;
  showLabel?: React.ReactNode;
  componentId?: string;
}

export const CopyButton = ({ copyText, showLabel = true, componentId, ...buttonProps }: CopyButtonProps) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const handleClick = () => {
    navigator.clipboard.writeText(copyText);
    setShowTooltip(true);
    setTimeout(() => {
      setShowTooltip(false);
    }, 3000);
  };

  const handleMouseLeave = () => {
    setShowTooltip(false);
  };

  return (
    <LegacyTooltip
      title={
        <FormattedMessage defaultMessage="Copied" description="Tooltip text shown when copy operation completes" />
      }
      dangerouslySetAntdProps={{
        visible: showTooltip,
      }}
    >
      <Button
        componentId={componentId ?? 'mlflow.shared.copy_button'}
        type="primary"
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
        css={{ 'z-index': 1 }}
        // Define children as a explicit prop so it can be easily overrideable
        children={
          showLabel ? <FormattedMessage defaultMessage="Copy" description="Button text for copy button" /> : undefined
        }
        {...buttonProps}
      />
    </LegacyTooltip>
  );
};
