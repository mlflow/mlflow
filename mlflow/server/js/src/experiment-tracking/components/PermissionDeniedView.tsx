/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
    <div className="center">
      <img style={{ height: 300, marginTop: 80 }} src={permissionDeniedLock} alt="permission denied" />
      <h1 style={{ paddingTop: 10 }}>Permission Denied</h1>
      <h2 data-testid="error-message" css={{ color: theme.colors.textSecondary }}>
        {errorMessage || defaultMessage}
      </h2>
    </div>
  );
}
