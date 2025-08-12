import React from 'react';
import permissionDeniedLock from '../../common/static/permission-denied-lock.svg';
import { useDesignSystemTheme } from '@databricks/design-system';

const defaultMessage = 'The current user does not have permission to view this page.';

type Props = {
  errorMessage?: string;
};

export function PermissionDeniedView({ errorMessage }: Props) {
  const { theme } = useDesignSystemTheme();
  return (
    <div className="mlflow-center">
      <img style={{ height: 300, marginTop: 80 }} src={permissionDeniedLock} alt="permission denied" />
      <h1 style={{ paddingTop: 10 }}>Permission Denied</h1>
      <h2 data-testid="mlflow-error-message" css={{ color: theme.colors.textSecondary }}>
        {errorMessage || defaultMessage}
      </h2>
    </div>
  );
}
