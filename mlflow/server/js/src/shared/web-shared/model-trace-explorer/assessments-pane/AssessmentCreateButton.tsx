import { useEffect, useRef, useState } from 'react';

import { Button, PlusIcon } from '@databricks/design-system';

import { AssessmentCreateForm } from './AssessmentCreateForm';

export const AssessmentCreateButton = ({
  title,
  assessmentName,
  spanId,
  traceId,
}: {
  title: React.ReactNode;
  assessmentName?: string;
  spanId?: string;
  traceId: string;
}) => {
  const [expanded, setExpanded] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (expanded && ref.current) {
      // scroll form into view after the form is expanded
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [expanded]);

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
          ref={ref}
          assessmentName={assessmentName}
          spanId={spanId}
          traceId={traceId}
          setExpanded={setExpanded}
        />
      )}
    </div>
  );
};
