import React, { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, type ButtonProps, Tooltip } from '@databricks/design-system';

interface CopyButtonProps extends Partial<ButtonProps> {
  copyText: string;
  showLabel?: React.ReactNode;
}

export const CopyButton = ({ copyText, showLabel = true, ...buttonProps }: CopyButtonProps) => {
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
    <Tooltip
      title={
        <FormattedMessage defaultMessage="Copied" description="Tooltip text shown when copy operation completes" />
      }
      dangerouslySetAntdProps={{
        visible: showTooltip,
      }}
    >
      <Button
        componentId="codegen_mlflow_app_src_shared_building_blocks_copybutton.tsx_35"
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
    </Tooltip>
  );
};
