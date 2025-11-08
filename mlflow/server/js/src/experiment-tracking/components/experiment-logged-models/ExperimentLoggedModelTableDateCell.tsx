import React from 'react';

import { TimeAgo } from '@databricks/web-shared/browse';

export const ExperimentLoggedModelTableDateCell = ({ value }: { value?: string | number | null }) => {
  const date = new Date(Number(value));

  if (isNaN(date as any)) {
    return null;
  }

  return <TimeAgo date={date} />;
};
