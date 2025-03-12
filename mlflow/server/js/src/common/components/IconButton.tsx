import React from 'react';
import { Button } from '@databricks/design-system';

type Props = {
  icon: React.ReactNode;
  style?: React.CSSProperties;
  className?: string;
  onClick?: () => void;
  restProps?: unknown;
};

export const IconButton = ({ icon, className, style, onClick, ...restProps }: Props) => {
  return (
    <Button
      componentId="codegen_mlflow_app_src_common_components_iconbutton.tsx_20"
      type="link"
      className={className}
      style={{ padding: 0, ...style }}
      onClick={onClick}
      {...restProps}
    >
      {icon}
    </Button>
  );
};
