/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
    <Button
      componentId="codegen_mlflow_app_src_common_components_iconbutton.tsx_20"
      type="link"
      className={className}
      style={{ padding: 0, ...style }}
      {...restProps}
    >
      {icon}
    </Button>
  );
};
