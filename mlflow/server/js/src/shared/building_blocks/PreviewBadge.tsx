import React from 'react';
import { FormattedMessage } from 'react-intl';

import { Tag, useDesignSystemTheme } from '@databricks/design-system';
export const PreviewBadge = ({ className }: { className?: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Tag className={className} css={{ marginLeft: theme.spacing.xs }} color="turquoise">
      <FormattedMessage
        defaultMessage="Experimental"
        description="Experimental badge shown for features which are experimental"
      />
    </Tag>
  );
};
