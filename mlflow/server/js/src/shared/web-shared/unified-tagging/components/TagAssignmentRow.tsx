import invariant from 'invariant';
import React from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

export function TagAssignmentRow({ children }: { children: React.ReactNode }) {
  const { theme } = useDesignSystemTheme();

  const stableChildren = React.Children.toArray(children);
  invariant(stableChildren.length <= 3, 'TagAssignmentRow must have 3 children or less');

  const parsedChildren = Array(3)
    .fill(null)
    .map((_, i) => stableChildren[i] ?? <span key={i} style={{ width: theme.general.heightSm }} />); // Sync width with only icon button width

  return (
    <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr min-content', gap: theme.spacing.sm }}>
      {parsedChildren}
    </div>
  );
}
