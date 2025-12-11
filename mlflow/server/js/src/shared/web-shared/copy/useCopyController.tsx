import type { ReactElement } from 'react';
import React, { useEffect, useRef, useState } from 'react';
import { useClipboard } from 'use-clipboard-copy';

import { CheckIcon, CopyIcon } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

export interface CopyController {
  actionIcon: ReactElement;
  tooltipMessage: string;
  copy: () => void;
  copied: boolean;
  ariaLabel: string;
  tooltipOpen: boolean;
  handleTooltipOpenChange: (open: boolean) => void;
}

/**
 * Utility hook that is internal to web-shared, use: `Copyable` or `CopyActionButton`
 *  or if it's a `CodeSnippet`, `SnippetCopyAction`
 */
export function useCopyController(text: string, copyTooltip?: string, onCopy?: () => void): CopyController {
  const intl = useIntl();

  const copyMessage = copyTooltip
    ? copyTooltip
    : intl.formatMessage({
        defaultMessage: 'Copy',
        description: 'Tooltip message displayed on copy action',
      });

  const copiedMessage = intl.formatMessage({
    defaultMessage: 'Copied',
    description: 'Tooltip message displayed on copy action after it has been clicked',
  });

  const clipboard = useClipboard();
  const copiedTimerIdRef = useRef<number>();
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    return () => {
      window.clearTimeout(copiedTimerIdRef.current);
    };
  }, []);

  const copy = () => {
    clipboard.copy(text);
    window.clearTimeout(copiedTimerIdRef.current);
    setCopied(true);
    onCopy?.();
    copiedTimerIdRef.current = window.setTimeout(() => {
      setCopied(false);
    }, 3000);
  };

  return {
    actionIcon: copied ? <CheckIcon /> : <CopyIcon />,
    tooltipMessage: copied ? copiedMessage : copyMessage,
    copy,
    copied,
    ariaLabel: copyMessage,
    tooltipOpen: open || copied,
    handleTooltipOpenChange: setOpen,
  };
}
