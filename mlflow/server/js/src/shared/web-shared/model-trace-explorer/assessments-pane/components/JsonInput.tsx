import { FormUI, Input } from '@databricks/design-system';

import type { AssessmentValueInputFieldProps } from './types';

export const JsonInput = ({
  value,
  valueError,
  setValue,
  setValueError,
  isSubmitting,
}: AssessmentValueInputFieldProps) => {
  return (
    <div>
      <Input.TextArea
        data-testid="assessment-value-json-input"
        componentId="shared.model-trace-explorer.assessment-edit-value-string-input"
        value={String(value)}
        autoSize={{ minRows: 1, maxRows: 5 }}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => {
          setValue(e.target.value);
          setValueError(null);
        }}
        validationState={valueError ? 'error' : undefined}
        disabled={isSubmitting}
      />
      {valueError && (
        <FormUI.Message
          id="shared.model-trace-explorer.assessment-edit-value-json-error"
          message={valueError}
          type="error"
        />
      )}
    </div>
  );
};
