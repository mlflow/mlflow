import { css } from '@emotion/react';
import type { ReactNode } from 'react';
import React from 'react';

import type { ButtonProps } from '@databricks/design-system';
import { Button, LegacyTooltip } from '@databricks/design-system';

type SnippetActionButtonProps = Pick<ButtonProps, 'icon' | 'onClick' | 'href' | 'rel' | 'target'> & {
  tooltipMessage: NonNullable<ReactNode>;
};

export default function SnippetActionButton({ tooltipMessage, ...buttonProps }: SnippetActionButtonProps) {
  const style = css({
    zIndex: 1, // required for action buttons to be visible and float
  });
  return (
    <LegacyTooltip title={tooltipMessage}>
      <Button
        componentId="codegen_web-shared_src_snippet_actions_snippetactionbutton.tsx_33"
        {...buttonProps}
        css={style}
      />
    </LegacyTooltip>
  );
}
