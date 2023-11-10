import React from 'react';
import { FormattedMessage } from 'react-intl';
import { Tag } from '@databricks/design-system';

type Props = {
  className?: string;
};

export const PreviewIcon = ({ className }: Props) => (
  <Tag style={{ marginLeft: '4px' }} color='turquoise' className={className}>
    <FormattedMessage
      defaultMessage='Preview'
      description='Preview badge shown for features which are under preview'
    />
  </Tag>
);
