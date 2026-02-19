/**
 * Component that renders labeling tasks for a trace
 *
 * Displays all labeling schemas for the current session and allows
 * the user to provide assessments (feedback/expectations).
 */

import React from 'react';
import {
  Accordion,
  Button,
  CircleIcon,
  CheckCircleIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import type { TaskRef } from './LabelingContextProvider';
import { useLabelingContext } from './LabelingContextProvider';
import { TextTask, NumericTask, CategoricalTask, CategoricalListTask, TextListTask } from './schemas';
import type { LabelingSession, LabelingSchema } from '../../../types/labeling';
import type { Assessment } from '../../../../shared/web-shared/model-trace-explorer/ModelTrace.types';
import { getAssessmentForLabelingSchema, isTaskCompleted } from '../../../utils/labeling';

interface LabelingTasksProps {
  session: LabelingSession;
  traceId: string;
  assessments: Assessment[];
}

export const LabelingTasks = ({ session, traceId, assessments }: LabelingTasksProps) => {
  const { activeTask, setActiveTask } = useLabelingContext();
  const { theme } = useDesignSystemTheme();

  const isActive = activeTask?.traceId === traceId;

  const expectationSchemas = session.labelingSchemas.filter((schema) => schema.type === 'EXPECTATION');
  const feedbackSchemas = session.labelingSchemas.filter((schema) => schema.type === 'FEEDBACK');

  return (
    <>
      {expectationSchemas.length > 0 && (
        <AccordionGroup
          title="Expectations"
          labelingSchemas={expectationSchemas}
          traceId={traceId}
          assessments={assessments}
          activeTask={activeTask}
          setActiveTask={setActiveTask}
          isActive={isActive}
        />
      )}
      {feedbackSchemas.length > 0 && (
        <AccordionGroup
          title="Feedback"
          labelingSchemas={feedbackSchemas}
          traceId={traceId}
          assessments={assessments}
          activeTask={activeTask}
          setActiveTask={setActiveTask}
          isActive={isActive}
        />
      )}
    </>
  );
};

interface AccordionGroupProps {
  title: string;
  labelingSchemas: LabelingSchema[];
  traceId: string;
  assessments: Assessment[];
  activeTask: TaskRef | null;
  setActiveTask: (task: TaskRef | null) => void;
  isActive: boolean;
}

const AccordionGroup = ({
  title,
  labelingSchemas,
  traceId,
  assessments,
  activeTask,
  setActiveTask,
  isActive,
}: AccordionGroupProps) => {
  const { theme } = useDesignSystemTheme();

  const taskRefToKey = (ref: TaskRef) => `${ref.traceId}__${ref.schemaName}`;
  const keyToTaskRef = (key: string): TaskRef => {
    const [traceId, schemaName] = key.split('__');
    return { traceId, schemaName };
  };

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.legacyBorders.borderRadiusLg,
        padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
        marginBottom: theme.spacing.md,
        boxShadow: isActive ? theme.shadows.xl : undefined,
      }}
    >
      <Typography.Title level={4}>{title}</Typography.Title>
      <Accordion
        activeKey={activeTask ? taskRefToKey(activeTask) : undefined}
        componentId="mlflow.labeling.tasks-accordion"
        displayMode="single"
        onChange={(key) => {
          if (!key) {
            setActiveTask(null);
          } else if (typeof key === 'string') {
            setActiveTask(keyToTaskRef(key));
          }
        }}
      >
        {labelingSchemas.map((schema) => (
          <SingleTask
            key={taskRefToKey({ traceId, schemaName: schema.name })}
            schema={schema}
            traceId={traceId}
            assessments={assessments}
          />
        ))}
      </Accordion>
    </div>
  );
};

interface SingleTaskProps {
  schema: LabelingSchema;
  traceId: string;
  assessments: Assessment[];
}

const SingleTask = ({ schema, traceId, assessments }: SingleTaskProps) => {
  const { theme } = useDesignSystemTheme();
  const taskDivRef = React.useRef<HTMLDivElement>(null);

  const { activeTask, updateProgress, goToNextUncompletedTask, setTaskDivRef } = useLabelingContext();

  // Find the assessment for this schema
  const currentAssessment = getAssessmentForLabelingSchema({ traceInfo: { trace_id: traceId } as any, assessments }, schema);
  const currentTaskIsCompleted = isTaskCompleted(currentAssessment, schema);

  // Update progress
  React.useEffect(() => {
    updateProgress({ traceId, schemaName: schema.name }, currentTaskIsCompleted);
  }, [traceId, schema.name, updateProgress, currentTaskIsCompleted]);

  // Register div ref for scrolling
  React.useEffect(() => {
    setTaskDivRef({ traceId, schemaName: schema.name }, taskDivRef);
  }, [traceId, schema.name, setTaskDivRef]);

  // State for the task value (local state before saving)
  const [value, setValue] = React.useState<any>(undefined);

  // Save handler - TODO: implement actual save to backend
  const saveValue = React.useCallback(
    (newValue: any) => {
      console.log('Saving assessment:', { schema: schema.name, value: newValue });
      // TODO: Call create/update assessment mutation
    },
    [schema.name],
  );

  // Render the appropriate task component based on schema type
  const renderTask = () => {
    switch (schema.schema.type) {
      case 'TEXT':
        return <TextTask schema={schema.schema} value={value} setValue={setValue} saveValue={saveValue} />;
      case 'NUMERIC':
        return <NumericTask schema={schema.schema} value={value} setValue={setValue} saveValue={saveValue} />;
      case 'CATEGORICAL':
        return <CategoricalTask schema={schema.schema} value={value} setValue={setValue} saveValue={saveValue} />;
      case 'CATEGORICAL_LIST':
        return <CategoricalListTask schema={schema.schema} value={value} setValue={setValue} saveValue={saveValue} />;
      case 'TEXT_LIST':
        return <TextListTask schema={schema.schema} value={value} setValue={setValue} saveValue={saveValue} />;
      default:
        return <div>Unsupported schema type</div>;
    }
  };

  return (
    <Accordion.Panel
      key={`${traceId}__${schema.name}`}
      header={
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {currentTaskIsCompleted ? (
            <CheckCircleIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />
          ) : (
            <CircleIcon css={{ color: theme.colors.textPlaceholder }} />
          )}
          <Typography.Text>{schema.title}</Typography.Text>
        </div>
      }
    >
      <div ref={taskDivRef} css={{ padding: theme.spacing.md }}>
        {schema.instructions && (
          <Typography.Paragraph css={{ marginBottom: theme.spacing.md }}>
            {schema.instructions}
          </Typography.Paragraph>
        )}
        {renderTask()}
        <div css={{ marginTop: theme.spacing.md, display: 'flex', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.labeling.task.next"
            type="primary"
            onClick={goToNextUncompletedTask}
          >
            Next
          </Button>
        </div>
      </div>
    </Accordion.Panel>
  );
};
