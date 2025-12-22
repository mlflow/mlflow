import {
  Button,
  DialogComboboxOptionList,
  DialogComboboxContent,
  DialogComboboxOptionListSelectItem,
} from '@databricks/design-system';

import { ChevronDownIcon, DialogCombobox, DialogComboboxCustomButtonTriggerWrapper } from '@databricks/design-system';
import { defineMessage, FormattedMessage } from 'react-intl';
import { EvaluateTracesParams } from './types';
import { isEmpty } from 'lodash';
import { useMemo, useState } from 'react';
import { coerceToEnum } from '../../../shared/web-shared/utils';

enum PickerOption {
  'LAST_TRACE' = '1',
  'LAST_5_TRACES' = '5',
  'LAST_10_TRACES' = '10',
  'CUSTOM' = 'custom',
}

const PickerOptionsLabels = {
  [PickerOption.LAST_TRACE]: defineMessage({
    defaultMessage: 'Last trace',
    description: 'Option for last trace',
  }),
  [PickerOption.LAST_5_TRACES]: defineMessage({
    defaultMessage: 'Last 5 traces',
    description: 'Option for last 5 traces',
  }),
  [PickerOption.LAST_10_TRACES]: defineMessage({
    defaultMessage: 'Last 10 traces',
    description: 'Option for last 10 traces',
  }),
  [PickerOption.CUSTOM]: defineMessage({
    defaultMessage: 'Select traces',
    description: 'Option for selecting custom traces',
  }),
};

export const SampleScorerTracesToEvaluatePicker = ({
  onTracesToEvaluateChange,
  tracesToEvaluate,
}: {
  tracesToEvaluate: Pick<EvaluateTracesParams, 'traceCount' | 'traceIds'>;
  onTracesToEvaluateChange: (tracesToEvaluate: Pick<EvaluateTracesParams, 'traceCount' | 'traceIds'>) => void;
}) => {
  const [displayPickCustomTracesModal, setDisplayPickCustomTracesModal] = useState(false);
  const tracesToEvaluateDropdownValue = useMemo(() => {
    if (!isEmpty(tracesToEvaluate.traceIds)) {
      return PickerOption.CUSTOM;
    }
    return coerceToEnum(PickerOption, String(tracesToEvaluate.traceCount), PickerOption.LAST_10_TRACES);
  }, [tracesToEvaluate]);

  return (
    <>
      <DialogCombobox
        componentId="mlflow.experiment-scorers.form.traces-picker"
        id="TODO"
        value={[tracesToEvaluateDropdownValue]}
      >
        <DialogComboboxCustomButtonTriggerWrapper>
          <Button
            componentId="mlflow.experiment-scorers.form.traces-picker.trigger"
            size="small"
            endIcon={<ChevronDownIcon />}
          >
            {tracesToEvaluateDropdownValue === PickerOption.CUSTOM ? (
              <FormattedMessage
                defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
                description="Label for the number of traces selected"
                values={{ count: tracesToEvaluate.traceIds?.length }}
              />
            ) : (
              <FormattedMessage {...PickerOptionsLabels[tracesToEvaluateDropdownValue]} />
            )}
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent align="end">
          <DialogComboboxOptionList>
            {[PickerOption.LAST_TRACE, PickerOption.LAST_5_TRACES, PickerOption.LAST_10_TRACES].map((value) => (
              <DialogComboboxOptionListSelectItem
                value={value}
                checked={tracesToEvaluateDropdownValue === value}
                onChange={() => {
                  onTracesToEvaluateChange({ traceCount: Number(value), traceIds: undefined });
                }}
              >
                <FormattedMessage {...PickerOptionsLabels[value]} />
              </DialogComboboxOptionListSelectItem>
            ))}
            {/* TODO(next PRs): Add custom trace selection option */}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
      {displayPickCustomTracesModal && (
        // TODO(next PRs): Add custom trace selection modal
        <div />
      )}
    </>
  );
};
