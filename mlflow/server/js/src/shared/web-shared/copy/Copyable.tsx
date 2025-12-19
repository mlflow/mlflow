import type { ReactNode } from 'react';
import React from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { CopyActionButton } from './CopyActionButton';

interface CommonProps {
  className?: string;
  /** Tracking id for copy icon button.
   * It is applied to its tooltip
   * appending "_tooltip" to it.*/
  componentId?: string;
  copyPlacement?: 'top' | 'center';
  copyTooltip?: string;
  onCopy?: () => void;
  stretch?: boolean;
}
interface WithCopyText extends CommonProps {
  children: ReactNode;
  copyText: string;
}
interface WithoutCopyText extends CommonProps {
  children: string;
  copyText?: null;
}
export type CopyableProps = WithCopyText | WithoutCopyText;

export function Copyable({
  children,
  componentId,
  copyPlacement = 'center',
  copyText,
  copyTooltip,
  onCopy,
  stretch = false,
}: CopyableProps) {
  const { theme } = useDesignSystemTheme();
  const copyTextOrChildren = copyText ?? children;
  return (
    <span css={{ display: 'flex', alignItems: 'stretch' }}>
      <span css={{ display: 'flex', alignItems: 'center', width: stretch ? '100%' : undefined, overflow: 'hidden' }}>
        <Typography.Text css={{ width: stretch ? '100%' : undefined, maxWidth: '100%' }}>{children}</Typography.Text>
      </span>
      <span
        css={{
          display: 'flex',
          alignItems: copyPlacement === 'center' ? 'center' : 'start',
          marginLeft: theme.spacing.sm,
        }}
      >
        <CopyActionButton
          componentId={componentId}
          copyText={copyTextOrChildren}
          copyTooltip={copyTooltip}
          onCopy={onCopy}
        />
      </span>
    </span>
  );
}
