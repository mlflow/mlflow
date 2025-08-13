import { FormattedMessage } from '@databricks/i18n';

import type { Expectation, Feedback } from '../ModelTrace.types';

export type AssessmentFormInputDataType = 'string' | 'boolean' | 'number' | 'json';

export const ASSESSMENT_PANE_MIN_WIDTH = 250;

// assessment names from databricks judges can sometimes have several
// prefixes that function like namespaces. for example:
//
// metric/global_guideline_adherence/api_code
//
// in this case, we only want to display the last element, as that
// is the most helpful name to the user (and to save ui space).
// if there are more slashes beyond that, we assume the user added
// it themselves, so we retain them.
export const getAssessmentDisplayName = (name: string): string => {
  const split = name.split('/');
  if (split.length === 1) {
    return name;
  } else if (split.length === 2) {
    return split[1];
  }
  return split.slice(2).join('/');
};

// forked from mlflow/web/js/src/common/utils/Utils.tsx
export const timeSinceStr = (date: any, referenceDate = new Date()) => {
  const seconds = Math.max(0, Math.floor((referenceDate.getTime() - date) / 1000));
  let interval = Math.floor(seconds / 31536000);

  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 year} other {# years}} ago"
        description="Text for time in years since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 2592000);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 month} other {# months}} ago"
        description="Text for time in months since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 86400);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 day} other {# days}} ago"
        description="Text for time in days since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 3600);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 hour} other {# hours}} ago"
        description="Text for time in hours since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 60);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 minute} other {# minutes}} ago"
        description="Text for time in minutes since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  return (
    <FormattedMessage
      defaultMessage="{timeSince, plural, =1 {1 second} other {# seconds}} ago"
      description="Text for time in seconds since given date for MLflow views"
      values={{ timeSince: seconds }}
    />
  );
};

export const getParsedExpectationValue = (expectation: Expectation) => {
  if ('value' in expectation) {
    return expectation.value;
  }

  try {
    // at the moment, "JSON_FORMAT" is the only serialization format
    // that is supported. in the future, we may switch on the
    // expectation.serialized_value.serialization_format field
    // to determine how to parse the value.
    return JSON.parse(expectation.serialized_value.value);
  } catch (e) {
    return expectation.serialized_value.value;
  }
};

export const getCreateAssessmentPayloadValue = ({
  formValue,
  dataType,
  isFeedback,
}: {
  formValue: string | boolean | number | null;
  dataType: AssessmentFormInputDataType;
  isFeedback: boolean;
}): { feedback: Feedback } | { expectation: Expectation } => {
  if (isFeedback) {
    return { feedback: { value: formValue } };
  }

  if (dataType === 'json') {
    return { expectation: { serialized_value: { value: String(formValue), serialization_format: 'JSON_FORMAT' } } };
  }

  return { expectation: { value: formValue } };
};
