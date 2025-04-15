import React from 'react';
import { FormattedMessage } from 'react-intl';

import { Tag, useDesignSystemTheme } from '@databricks/design-system';
export const FeatureBadge = ({
  type,
  expirationDate,
  className,
}: {
  type: 'new' | 'preview' | 'beta';
  expirationDate: Date;
  className?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Tag
      componentId="codegen_mlflow_app_src_shared_building_blocks_previewbadge.tsx_14"
      className={className}
      css={{ marginLeft: theme.spacing.xs }}
      color="turquoise"
    >
      <FormattedMessage
        defaultMessage="Experimental"
        description="Experimental badge shown for features which are experimental"
      />
    </Tag>
  );
};
