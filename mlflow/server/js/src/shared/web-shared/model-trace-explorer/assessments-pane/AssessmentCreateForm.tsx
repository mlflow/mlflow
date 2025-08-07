import { isNil } from 'lodash';
import { useCallback, useState } from 'react';

import {
  Button,
  FormUI,
  Input,
  SegmentedControlButton,
  SegmentedControlGroup,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import type { AssessmentFormInputDataType } from './AssessmentsPane.utils';
import { getCreateAssessmentPayloadValue } from './AssessmentsPane.utils';
import { displayErrorNotification, FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import type { CreateAssessmentPayload } from '../api';
import { createAssessment } from '../api';

export const AssessmentCreateForm = ({
  assessmentName,
  spanId,
  traceId,
  // used to close the form
  // after the assessment is created
  setExpanded,
}: {
  assessmentName?: string;
  spanId?: string;
  traceId: string;
  setExpanded: (expanded: boolean) => void;
}) => {
  const intl = useIntl();
  const queryClient = useQueryClient();
  const { theme } = useDesignSystemTheme();

  const [name, setName] = useState('');
  const [assessmentType, setAssessmentType] = useState<'feedback' | 'expectation'>('feedback');
  const [dataType, setDataType] = useState<AssessmentFormInputDataType>('boolean');
  const [value, setValue] = useState<string | boolean | number>(true);
  const [rationale, setRationale] = useState('');
  const [valueError, setValueError] = useState<React.ReactNode | null>(null);

  const { mutate: createAssessmentMutation, isLoading } = useMutation({
    mutationFn: (payload: CreateAssessmentPayload) => createAssessment({ payload }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
    },
    onError: (error) => {
      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to create assessment. Error: {error}',
            description: 'Error message when creating an assessment fails',
          },
          {
            error: error instanceof Error ? error.message : String(error),
          },
        ),
      );
    },
    onSettled: () => {
      setExpanded(false);
    },
  });

  const isNamePrefilled = !isNil(assessmentName);

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

  return (
    <div
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
        value={assessmentType}
        disabled={isLoading}
        onChange={(e) => setAssessmentType(e.target.value as 'feedback' | 'expectation')}
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
        <Input
          componentId="shared.model-trace-explorer.assessment-name-input"
          value={name}
          disabled={isLoading}
          onKeyDown={(e) => e.stopPropagation()}
          onChange={(e) => setName(e.target.value)}
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
          <FormattedMessage defaultMessage="String" description="String select menu option for assessment data type" />
        </SimpleSelectOption>
        <SimpleSelectOption value="boolean">
          <FormattedMessage
            defaultMessage="Boolean"
            description="Boolean select menu option for assessment data type"
          />
        </SimpleSelectOption>
        <SimpleSelectOption value="number">
          <FormattedMessage defaultMessage="Number" description="Numeric select menu option for assessment data type" />
        </SimpleSelectOption>
      </SimpleSelect>
      <Typography.Text css={{ marginTop: theme.spacing.xs }} size="sm" color="secondary">
        <FormattedMessage defaultMessage="Value" description="Field label for assessment value in a creation form" />
      </Typography.Text>
      {dataType === 'json' && (
        <>
          <Input.TextArea
            componentId="shared.model-trace-explorer.assessment-edit-value-string-input"
            value={String(value)}
            autoSize={{ minRows: 1, maxRows: 5 }}
            onKeyDown={(e) => e.stopPropagation()}
            onChange={(e) => {
              setValue(e.target.value);
              setValueError(null);
            }}
            validationState={valueError ? 'error' : undefined}
            disabled={isLoading}
          />
          {valueError && (
            <FormUI.Message
              id="shared.model-trace-explorer.assessment-edit-value-json-error"
              message={valueError}
              type="error"
            />
          )}
        </>
      )}
      {dataType === 'string' && (
        <Input
          componentId="shared.model-trace-explorer.assessment-value-string-input"
          value={String(value)}
          onKeyDown={(e) => e.stopPropagation()}
          onChange={(e) => {
            setValue(e.target.value);
            setValueError(null);
          }}
          disabled={isLoading}
          allowClear
        />
      )}
      {dataType === 'boolean' && (
        <SegmentedControlGroup
          componentId="shared.model-trace-explorer.assessment-value-boolean-input"
          name="shared.model-trace-explorer.assessment-value-boolean-input"
          value={value}
          disabled={isLoading}
          onChange={(e) => {
            setValue(e.target.value);
            setValueError(null);
          }}
        >
          <SegmentedControlButton value>True</SegmentedControlButton>
          <SegmentedControlButton value={false}>False</SegmentedControlButton>
        </SegmentedControlGroup>
      )}
      {dataType === 'number' && (
        <Input
          componentId="shared.model-trace-explorer.assessment-value-number-input"
          value={String(value)}
          onKeyDown={(e) => e.stopPropagation()}
          onChange={(e) => {
            setValue(e.target.value ? Number(e.target.value) : '');
            setValueError(null);
          }}
          type="number"
          disabled={isLoading}
          allowClear
        />
      )}
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
};
