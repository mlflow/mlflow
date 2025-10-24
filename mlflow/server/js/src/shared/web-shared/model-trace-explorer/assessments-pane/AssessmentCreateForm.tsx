import { isNil } from 'lodash';
import { forwardRef, useCallback, useState } from 'react';

import {
  Button,
  Input,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';

import { AssessmentCreateNameTypeahead } from './AssessmentCreateNameTypeahead';
import type { AssessmentFormInputDataType } from './AssessmentsPane.utils';
import { getCreateAssessmentPayloadValue } from './AssessmentsPane.utils';
import { BooleanInput } from './components/BooleanInput';
import { JsonInput } from './components/JsonInput';
import { NumericInput } from './components/NumericInput';
import { TextInput } from './components/TextInput';
import type { AssessmentValueInputFieldProps } from './components/types';
import type { CreateAssessmentPayload } from '../api';
import type { AssessmentSchema } from '../contexts/AssessmentSchemaContext';
import { useAssessmentSchemas } from '../contexts/AssessmentSchemaContext';
import { useCreateAssessment } from '../hooks/useCreateAssessment';

const ComponentMap: Record<AssessmentFormInputDataType, React.ComponentType<AssessmentValueInputFieldProps>> = {
  json: JsonInput,
  string: TextInput,
  boolean: BooleanInput,
  number: NumericInput,
};

type AssessmentCreateFormProps = {
  assessmentName?: string;
  spanId?: string;
  traceId: string;
  setExpanded: (expanded: boolean) => void;
};

export const AssessmentCreateForm = forwardRef<HTMLDivElement, AssessmentCreateFormProps>(
  (
    {
      assessmentName,
      spanId,
      traceId,
      // used to close the form
      // after the assessment is created
      setExpanded,
    },
    ref,
  ) => {
    const { theme } = useDesignSystemTheme();
    const { schemas } = useAssessmentSchemas();

    const [name, setName] = useState('');
    const [assessmentType, setAssessmentType] = useState<'feedback' | 'expectation'>('feedback');
    const [dataType, setDataType] = useState<AssessmentFormInputDataType>('boolean');
    const [value, setValue] = useState<string | boolean | number>(true);
    const [rationale, setRationale] = useState('');
    const [nameError, setNameError] = useState<React.ReactNode | null>(null);
    const [valueError, setValueError] = useState<React.ReactNode | null>(null);
    const isNamePrefilled = !isNil(assessmentName);

    // default to string if somehow the data type is not supported
    const InputComponent = ComponentMap[dataType] ?? ComponentMap['string'];

    const { createAssessmentMutation, isLoading } = useCreateAssessment({
      traceId,
      onSettled: () => {
        setExpanded(false);
      },
    });

    const handleCreate = useCallback(async () => {
      if (dataType === 'json') {
        try {
          // validate JSON
          JSON.parse(value as string);
        } catch (e) {
          setValueError(
            <FormattedMessage
              defaultMessage="The provided value is not valid JSON"
              description="Error message for invalid JSON in an assessment creation form"
            />,
          );
          return;
        }
      }

      if (!isNamePrefilled && name === '') {
        setNameError(
          <FormattedMessage
            defaultMessage="Please enter a name"
            description="Error message for empty assessment name in a creation form"
          />,
        );
        return;
      }

      const valueObj = getCreateAssessmentPayloadValue({
        formValue: value,
        dataType,
        isFeedback: assessmentType === 'feedback',
      });

      const payload: CreateAssessmentPayload = {
        assessment: {
          assessment_name: isNamePrefilled ? assessmentName : name,
          trace_id: traceId,
          source: {
            source_type: 'HUMAN',
            source_id: getUser() ?? '',
          },
          span_id: spanId,
          rationale,
          ...valueObj,
        },
      };

      createAssessmentMutation(payload);
    }, [
      dataType,
      value,
      assessmentType,
      isNamePrefilled,
      assessmentName,
      name,
      traceId,
      spanId,
      rationale,
      createAssessmentMutation,
    ]);

    const handleChangeSchema = useCallback(
      (schema: AssessmentSchema | null) => {
        // clear the form back to defaults
        if (!schema) {
          setName('');
          setAssessmentType('feedback');
          setDataType('boolean');
          setValue(true);
          setRationale('');
          setValueError(null);
          return;
        }

        // Check if this is a real schema from the schemas list or a fake one created for a new name
        const isRealSchema = schemas.some((s) => s.name === schema.name);

        // Only update the name if it's a new assessment name (not in schemas)
        // This preserves the user's selections for assessment type and data type
        if (!isRealSchema) {
          setName(schema.name);
          return;
        }

        // For existing schemas, update all fields
        setName(schema.name);
        setAssessmentType(schema.assessmentType);
        setDataType(schema.dataType);

        // set the appropriate empty value for the data type
        switch (schema.dataType) {
          case 'string':
          case 'json':
            setValue('');
            break;
          case 'number':
            setValue(0);
            break;
          case 'boolean':
            setValue(true);
            break;
        }
      },
      [schemas],
    );

    return (
      <div
        ref={ref}
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          marginTop: theme.spacing.sm,
          border: `1px solid ${theme.colors.border}`,
          padding: theme.spacing.sm,
          borderRadius: theme.borders.borderRadiusSm,
        }}
      >
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Assessment Type"
            description="Field label for assessment type in a creation form"
          />
        </Typography.Text>
        <SimpleSelect
          id="shared.model-trace-explorer.assessment-type-select"
          componentId="shared.model-trace-explorer.assessment-type-select"
          label="Assessment Type"
          value={assessmentType}
          disabled={isLoading}
          onChange={(e) => {
            setAssessmentType(e.target.value as 'feedback' | 'expectation');
            // JSON data is not available for feedback
            if (e.target.value === 'feedback' && dataType === 'json') {
              setDataType('string');
            }
          }}
        >
          <SimpleSelectOption value="feedback">
            <FormattedMessage defaultMessage="Feedback" description="Feedback select menu option for assessment type" />
          </SimpleSelectOption>
          <SimpleSelectOption value="expectation">
            <FormattedMessage
              defaultMessage="Expectation"
              description="Expectation select menu option for assessment type"
            />
          </SimpleSelectOption>
        </SimpleSelect>
        <Typography.Text css={{ marginTop: theme.spacing.xs }} size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Assessment Name"
            description="Field label for assessment name in a creation form"
          />
        </Typography.Text>
        {isNamePrefilled ? (
          <Typography.Text>{assessmentName}</Typography.Text>
        ) : (
          <AssessmentCreateNameTypeahead
            name={name}
            setName={setName}
            handleChangeSchema={handleChangeSchema}
            nameError={nameError}
            setNameError={setNameError}
          />
        )}
        <Typography.Text css={{ marginTop: theme.spacing.xs }} size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Data Type"
            description="Field label for assessment data type in a creation form"
          />
        </Typography.Text>
        <SimpleSelect
          id="shared.model-trace-explorer.assessment-data-type-select"
          componentId="shared.model-trace-explorer.assessment-data-type-select"
          label="Data Type"
          value={dataType}
          disabled={isLoading}
          onChange={(e) => {
            setDataType(e.target.value as AssessmentFormInputDataType);
            setValueError(null);
          }}
        >
          {assessmentType === 'expectation' && (
            <SimpleSelectOption value="json">
              <FormattedMessage defaultMessage="JSON" description="JSON select menu option for assessment data type" />
            </SimpleSelectOption>
          )}
          <SimpleSelectOption value="string">
            <FormattedMessage
              defaultMessage="String"
              description="String select menu option for assessment data type"
            />
          </SimpleSelectOption>
          <SimpleSelectOption value="boolean">
            <FormattedMessage
              defaultMessage="Boolean"
              description="Boolean select menu option for assessment data type"
            />
          </SimpleSelectOption>
          <SimpleSelectOption value="number">
            <FormattedMessage
              defaultMessage="Number"
              description="Numeric select menu option for assessment data type"
            />
          </SimpleSelectOption>
        </SimpleSelect>
        <Typography.Text css={{ marginTop: theme.spacing.xs }} size="sm" color="secondary">
          <FormattedMessage defaultMessage="Value" description="Field label for assessment value in a creation form" />
        </Typography.Text>
        <InputComponent
          value={value}
          valueError={valueError}
          setValue={setValue}
          setValueError={setValueError}
          isSubmitting={isLoading}
        />
        <Typography.Text css={{ marginTop: theme.spacing.xs }} size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Rationale"
            description="Field label for assessment rationale in a creation form"
          />
        </Typography.Text>
        <Input.TextArea
          componentId="shared.model-trace-explorer.assessment-rationale-input"
          value={rationale}
          autoSize={{ minRows: 1, maxRows: 5 }}
          disabled={isLoading}
          onKeyDown={(e) => e.stopPropagation()}
          onChange={(e) => setRationale(e.target.value)}
        />
        <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.xs }}>
          <Button
            componentId="shared.model-trace-explorer.assessment-create-button"
            disabled={isLoading}
            onClick={() => setExpanded(false)}
          >
            <FormattedMessage
              defaultMessage="Cancel"
              description="Button label for cancelling the creation of an assessment"
            />
          </Button>
          <Button
            css={{ marginLeft: theme.spacing.sm }}
            type="primary"
            componentId="shared.model-trace-explorer.assessment-create-button"
            onClick={handleCreate}
            loading={isLoading}
          >
            <FormattedMessage defaultMessage="Create" description="Button label for creating an assessment" />
          </Button>
        </div>
      </div>
    );
  },
);
