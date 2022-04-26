import React from 'react';
import {
  CircleIcon as DuboisCircleIcon,
  CheckCircleBorderIcon,
  useDesignSystemTheme,
  WarningFillIcon,
} from '@databricks/design-system';
import { css } from 'emotion';
import { PropTypes } from 'prop-types';
/**
 * Get a unique key for a model version object.
 * @param modelName
 * @param version
 * @returns {string}
 */
export const getModelVersionKey = (modelName, version) => `${modelName}_${version}`;

export const getProtoField = (fieldName) => `${fieldName}`;

export function ReadyIcon() {
  const { theme } = useDesignSystemTheme();
  return <CheckCircleBorderIcon className={css({ color: theme.colors.textValidationSuccess })} />;
}

export function FailedIcon() {
  const { theme } = useDesignSystemTheme();
  return <WarningFillIcon className={css({ color: theme.colors.textValidationDanger })} />;
}

export function CircleIcon({ type }) {
  const { theme } = useDesignSystemTheme();
  let color;
  switch (type) {
    case 'FAILED': {
      color = theme.colors.textValidationDanger;
      break;
    }
    case 'PENDING': {
      color = theme.colors.yellow500; // textValidationWarning was too dark/red
      break;
    }
    case 'READY':
    default: {
      color = theme.colors.green500;
      break;
    }
  }
  return <DuboisCircleIcon className={css({ color })} />;
}
CircleIcon.propTypes = {
  type: PropTypes.oneOf(['FAILED', 'PENDING', 'READY']).isRequired,
};
