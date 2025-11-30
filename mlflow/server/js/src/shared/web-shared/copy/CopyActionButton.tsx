import React from 'react';

import type { ButtonProps, TooltipProps } from '@databricks/design-system';
import { Button, Tooltip } from '@databricks/design-system';

import { useCopyController } from './useCopyController';

export interface CopyActionButtonProps {
  buttonProps?: Partial<ButtonProps>;
  componentId?: string;
  copyText: string;
  copyTooltip?: string;
  isInsideInputGroup?: boolean;
  onCopy?: () => void;
  tooltipProps?: Partial<TooltipProps>;
}

export function CopyActionButton({
  buttonProps,
  componentId,
  copyText,
  copyTooltip,
  isInsideInputGroup = false,
  onCopy,
  tooltipProps,
}: CopyActionButtonProps) {
  const { actionIcon, copy, handleTooltipOpenChange, tooltipOpen, tooltipMessage } = useCopyController(
    copyText,
    copyTooltip,
    onCopy,
  );

  const button = (
    <Button
      aria-label={tooltipMessage}
      componentId={componentId ?? 'codegen_web-shared_src_copy_copyactionbutton.tsx_17'}
      icon={actionIcon}
      onClick={copy}
      size="small"
      {...buttonProps}
    />
  );

  const inputGroupButton = (
    <Button
      aria-label={tooltipMessage}
      componentId={componentId ?? 'codegen_web-shared_src_copy_copyactionbutton.tsx_17'}
      onClick={copy}
      {...buttonProps}
    >
      {actionIcon}
    </Button>
  );

  return (
    <Tooltip
      componentId={
        componentId ? `${componentId}-tooltip` : 'codegen_web-shared_src_copy_copyactionbutton.tsx_17-tooltip'
      }
      content={tooltipMessage}
      onOpenChange={handleTooltipOpenChange}
      open={tooltipOpen}
      {...tooltipProps}
    >
      {isInsideInputGroup ? inputGroupButton : button}
    </Tooltip>
  );
}
