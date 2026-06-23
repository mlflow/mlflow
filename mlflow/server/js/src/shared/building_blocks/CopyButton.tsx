import React, { useEffect, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, type ButtonProps, Tooltip } from '@databricks/design-system';
import { copyToClipboard } from '../../common/utils/copyToClipboard';

interface CopyButtonProps extends Partial<ButtonProps> {
  copyText: string;
  showLabel?: React.ReactNode;
  componentId?: string;
}

export const CopyButton = ({ copyText, showLabel = true, componentId, ...buttonProps }: CopyButtonProps) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [showCopyError, setShowCopyError] = useState(false);
  const tooltipTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const errorTimerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(
    () => () => {
      clearTimeout(tooltipTimerRef.current);
      clearTimeout(errorTimerRef.current);
    },
    [],
  );

  const handleClick = async () => {
    const success = await copyToClipboard(copyText);
    if (success) {
      setShowTooltip(true);
      clearTimeout(tooltipTimerRef.current);
      tooltipTimerRef.current = setTimeout(() => setShowTooltip(false), 3000);
    } else {
      setShowCopyError(true);
      clearTimeout(errorTimerRef.current);
      errorTimerRef.current = setTimeout(() => setShowCopyError(false), 3000);
    }
  };

  const handleMouseLeave = () => {
    setShowTooltip(false);
    setShowCopyError(false);
  };

  return (
    <Tooltip
      content={
        showCopyError ? (
          <FormattedMessage
            defaultMessage="Copy failed. Clipboard unavailable."
            description="Tooltip text shown when copy operation fails"
          />
        ) : (
          <FormattedMessage defaultMessage="Copied" description="Tooltip text shown when copy operation completes" />
        )
      }
      open={showTooltip || showCopyError}
      componentId="mlflow.shared.copy_button.tooltip"
    >
      <Button
        componentId={componentId ?? 'mlflow.shared.copy_button'}
        type="primary"
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
        css={{ 'z-index': 1 }}
        children={
          showLabel ? <FormattedMessage defaultMessage="Copy" description="Button text for copy button" /> : undefined
        }
        {...buttonProps}
      />
    </Tooltip>
  );
};
