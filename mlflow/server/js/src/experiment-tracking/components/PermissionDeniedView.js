import React from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';
import permissionDeniedLock from '../../common/static/permission-denied-lock.svg';
import { useDesignSystemTheme } from '@databricks/design-system';

const defaultMessage = 'The current user does not have permission to view this page.';

export function PermissionDeniedView({ errorMessage }) {
  const { theme } = useDesignSystemTheme();
  return (
    <div className='center'>
      <img
        style={{ height: 300, marginTop: 80 }}
        src={permissionDeniedLock}
        alt='permission denied'
      />
      <h1 style={{ paddingTop: 10 }}>Permission Denied</h1>
      <h2 data-testid='error-message' className={css({ color: theme.colors.textSecondary })}>
        {errorMessage || defaultMessage}
      </h2>
    </div>
  );
}

PermissionDeniedView.propTypes = {
  errorMessage: PropTypes.string,
};
