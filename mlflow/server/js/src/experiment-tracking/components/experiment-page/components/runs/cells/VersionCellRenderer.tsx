import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import { RunRowVersionInfo } from '../../../utils/experimentPage.row-types';

export const VersionCellRenderer = React.memo(
  ({
    value: {
      // Run row version object parameters
      version,
      name,
      type,
    },
  }: {
    value: RunRowVersionInfo;
  }) =>
    Utils.renderSourceVersion(
      // Using function from utils to render the source link
      version,
      name,
      type,
    ) || <>-</>,
);
