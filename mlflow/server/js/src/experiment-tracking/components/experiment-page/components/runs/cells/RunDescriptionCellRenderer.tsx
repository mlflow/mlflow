import React from 'react';
import { Tooltip } from '@databricks/design-system';
import Utils from '../../../../../../common/utils/Utils';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import { TrimmedText } from '../../../../../../common/components/TrimmedText';

export const RunDescriptionCellRenderer = React.memo(({ value }: { value: RunRowType['tags'] }) => {
  const description = Utils.getRunDescriptionFromTags(value) || '-';
  return (
    <>
      <Tooltip componentId="mlflow.experiment-tracking.run-description.display" content={description}>
        <span>
          <TrimmedText text={description} maxSize={50} />
        </span>
      </Tooltip>
    </>
  );
});
