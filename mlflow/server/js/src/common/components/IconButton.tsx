import React from 'react';
import { Button } from '@databricks/design-system';

type Props = {
  icon: React.ReactNode;
  style?: any;
  className?: string;
  restProps?: any;
};

export const IconButton = ({ icon, className, style, ...restProps }: Props) => {
  return (
    <Button type='link' className={className} style={{ padding: 0, ...style }} {...restProps}>
      {icon}
    </Button>
  );
};
