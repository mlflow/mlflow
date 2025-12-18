import { useEffect, useRef, useState } from 'react';

import { Button, PlusIcon } from '@databricks/design-system';

import { AssessmentCreateForm } from './AssessmentCreateForm';

export const AssessmentCreateButton = ({
  title,
  assessmentName,
  spanId,
  traceId,
  defaultMetadata,
}: {
  title: React.ReactNode;
  assessmentName?: string;
  spanId?: string;
  traceId: string;
  defaultMetadata?: Record<string, string>;
}) => {
  const [expanded, setExpanded] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (expanded && ref.current) {
      // scroll form into view after the form is expanded, but only if it's not already visible
      const element = ref.current;
      const parent = element.offsetParent as HTMLElement;
      if (parent) {
        const elementTop = element.offsetTop;
        const parentScrollTop = parent.scrollTop;
        const parentHeight = parent.clientHeight;

        // Only scroll if the element is not fully visible
        if (elementTop < parentScrollTop || elementTop > parentScrollTop + parentHeight) {
          ref.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
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
          defaultMetadata={defaultMetadata}
        />
      )}
    </div>
  );
};
