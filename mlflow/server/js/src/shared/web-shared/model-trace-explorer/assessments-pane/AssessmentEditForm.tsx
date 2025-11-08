import { useCallback, useState } from 'react';

import {
  Typography,
  useDesignSystemTheme,
  SimpleSelect,
  SimpleSelectOption,
  SegmentedControlGroup,
  SegmentedControlButton,
  Input,
  Button,
  FormUI,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';

import type { AssessmentFormInputDataType } from './AssessmentsPane.utils';
import { getCreateAssessmentPayloadValue } from './AssessmentsPane.utils';
import { getAssessmentValue } from './utils';
import type { Assessment } from '../ModelTrace.types';
import type { UpdateAssessmentPayload } from '../api';
import { useOverrideAssessment } from '../hooks/useOverrideAssessment';
import { useUpdateAssessment } from '../hooks/useUpdateAssessment';

// default to the original type of the value if possible. however,
// we only support editing simple types in the UI (i.e. not arrays / objects)
// so if the value does not fit, we just default to boolean for simplicity
const getDefaultType = (value: any, isFeedback: boolean): AssessmentFormInputDataType => {
  if (typeof value === 'string') {
    // treat empty strings as null, default to boolean
    if (value === '') {
      return 'boolean';
    }

    if (isFeedback) {
      return 'string';
    }

    try {
      JSON.parse(value);
      return 'json';
    } catch (e) {
      // not valid JSON, default to string
      return 'string';
    }
  }

  if (typeof value === 'boolean' || typeof value === 'number') {
    return typeof value as 'boolean' | 'number';
  }
  return 'boolean';
};

const getDefaultValue = (value: any): string | boolean | number | null => {
  if (typeof value === 'string') {
    // treat empty strings as null
    return value || null;
  }
  if (typeof value === 'boolean' || typeof value === 'number') {
    return value;
  }
  return null;
};

export const AssessmentEditForm = ({
  assessment,
  onSuccess,
  onSettled,
  onCancel,
}: {
  assessment: Assessment;
  onSuccess?: () => void;
  onSettled?: () => void;
  onCancel: () => void;
}) => {
  const isFeedback = 'feedback' in assessment;
  const initialValue = getAssessmentValue(assessment);
  const defaultType = getDefaultType(initialValue, isFeedback);
  const defaultValue = getDefaultValue(initialValue);
  const user = getUser() ?? '';

  const { theme } = useDesignSystemTheme();
  const [dataType, setDataType] = useState<AssessmentFormInputDataType>(defaultType);
  const [value, setValue] = useState<string | boolean | number | null>(defaultValue);
  const [rationale, setRationale] = useState(assessment.rationale);
  const [valueError, setValueError] = useState<React.ReactNode | null>(null);

  const { updateAssessmentMutation, isLoading: isUpdating } = useUpdateAssessment({
    assessment,
    onSuccess,
    onSettled,
  });

  const { overrideAssessmentMutation, isLoading: isOverwriting } = useOverrideAssessment({
    traceId: assessment.trace_id,
    onSuccess,
    onSettled,
  });

  const isLoading = isUpdating || isOverwriting;

  const handleUpdate = useCallback(async () => {
    if (dataType === 'json') {
      try {
        JSON.parse(value as string);
      } catch (e) {
        setValueError(
          <FormattedMessage
            defaultMessage="The provided value is not valid JSON"
            description="Error message for invalid JSON in an assessment edit form"
          />,
        );
        return;
      }
    }

    const valueObj = getCreateAssessmentPayloadValue({
      formValue: value,
      dataType,
      isFeedback,
    });

    // if a user edits their own assessment, we update it in
    // place as they are likely just correcting a mistake.
    // expectation edits should always call the update API
    if (user === assessment.source.source_id || !isFeedback) {
      const payload: UpdateAssessmentPayload = {
        assessment: {
          ...valueObj,
          rationale,
        },
        update_mask: `${isFeedback ? 'feedback' : 'expectation'},rationale`,
      };

      updateAssessmentMutation(payload);
    } else {
      overrideAssessmentMutation({
        oldAssessment: assessment,
        value: valueObj,
        ...(rationale ? { rationale } : {}),
      });
    }
  }, [dataType, value, isFeedback, user, assessment, rationale, updateAssessmentMutation, overrideAssessmentMutation]);

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
      <Typography.Text css={{ marginTop: theme.spacing.xs }} size="sm" color="secondary">
        <FormattedMessage
          defaultMessage="Data Type"
          description="Field label for assessment data type in an edit form"
        />
      </Typography.Text>
      <SimpleSelect
        id="shared.model-trace-explorer.assessment-edit-data-type-select"
        componentId="shared.model-trace-explorer.assessment-edit-data-type-select"
        value={dataType}
        disabled={isLoading}
        onChange={(e) => {
          setDataType(e.target.value as AssessmentFormInputDataType);
          setValueError(null);
        }}
      >
        {!isFeedback && (
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
        <FormattedMessage defaultMessage="Value" description="Field label for assessment value in an edit form" />
      </Typography.Text>
      {dataType === 'json' && (
        <>
          <Input.TextArea
            componentId="shared.model-trace-explorer.assessment-edit-value-string-input"
            value={String(value)}
            rows={3}
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
          componentId="shared.model-trace-explorer.assessment-edit-value-string-input"
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
          componentId="shared.model-trace-explorer.assessment-edit-value-boolean-input"
          name="shared.model-trace-explorer.assessment-edit-value-boolean-input"
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
          componentId="shared.model-trace-explorer.assessment-edit-value-number-input"
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
          description="Field label for assessment rationale in an edit form"
        />
      </Typography.Text>
      <Input.TextArea
        componentId="shared.model-trace-explorer.assessment-edit-rationale-input"
        value={rationale}
        autoSize={{ minRows: 1, maxRows: 5 }}
        disabled={isLoading}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => setRationale(e.target.value)}
      />
      <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.xs }}>
        <Button
          componentId="shared.model-trace-explorer.assessment-edit-cancel-button"
          disabled={isLoading}
          onClick={onCancel}
        >
          <FormattedMessage
            defaultMessage="Cancel"
            description="Button label for cancelling the edit of an assessment"
          />
        </Button>
        <Button
          css={{ marginLeft: theme.spacing.sm }}
          type="primary"
          componentId="shared.model-trace-explorer.assessment-edit-save-button"
          onClick={handleUpdate}
          loading={isLoading}
        >
          <FormattedMessage defaultMessage="Save" description="Button label for saving an edit to an assessment" />
        </Button>
      </div>
    </div>
  );
};
