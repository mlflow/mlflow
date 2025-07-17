import React from 'react';

import type { ButtonProps } from '@databricks/design-system';

import SnippetActionButton from './SnippetActionButton';
import { useCopyController } from '../hooks/useCopyController';

export interface SnippetCopyActionProps extends ButtonProps {
  /**
   * The text to be copied into clipboard when action button is clicked.
   */
  copyText: string;
  onClick?: (e: React.MouseEvent) => void;
}

export function SnippetCopyAction({ copyText, onClick, ...props }: SnippetCopyActionProps) {
  const { actionIcon, tooltipMessage, copy } = useCopyController(copyText);

  return (
    <SnippetActionButton
      tooltipMessage={tooltipMessage}
      icon={actionIcon}
      onClick={(e) => {
        copy();
        onClick?.(e);
      }}
      {...props}
    />
  );
}
