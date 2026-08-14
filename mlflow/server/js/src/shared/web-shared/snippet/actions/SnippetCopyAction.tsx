import React from 'react';

import type { ButtonProps } from '@databricks/design-system';
import { useCopyController } from '@databricks/web-shared/copy';

import SnippetActionButton from './SnippetActionButton';

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
