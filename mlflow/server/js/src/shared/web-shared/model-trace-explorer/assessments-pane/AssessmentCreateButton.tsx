import { useState } from 'react';

import { Button, PlusIcon } from '@databricks/design-system';

import { AssessmentCreateForm } from './AssessmentCreateForm';

export const AssessmentCreateButton = ({
  title,
  assessmentName,
  spanId,
  traceId,
}: {
  title: string;
  assessmentName?: string;
  spanId?: string;
  traceId: string;
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div>
      <Button
        size="small"
        componentId="shared.model-trace-explorer.add-new-assessment"
        icon={<PlusIcon />}
        onClick={() => setExpanded(true)}
      >
        {title}
      </Button>
      {expanded && (
        <AssessmentCreateForm
          assessmentName={assessmentName}
          spanId={spanId}
          traceId={traceId}
          setExpanded={setExpanded}
        />
      )}
    </div>
  );
};
