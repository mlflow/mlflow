import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import type { RunRowVersionInfo } from '../../../utils/experimentPage.row-types';

export const VersionCellRenderer = React.memo(({ value }: { value?: RunRowVersionInfo }) => {
  if (!value) {
    return <>-</>;
  }
  const {
    // Run row version object parameters
    version,
    name,
    type,
  } = value;

  return (
    Utils.renderSourceVersion(
      // Using function from utils to render the source link
      version,
      name,
      type,
    ) || <>-</>
  );
});
