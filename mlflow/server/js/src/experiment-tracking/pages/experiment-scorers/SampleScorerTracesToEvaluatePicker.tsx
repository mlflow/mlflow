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
import { SelectTracesModal } from '../../components/SelectTracesModal';

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
  onItemsToEvaluateChange,
  itemsToEvaluate,
}: {
  itemsToEvaluate: Pick<EvaluateTracesParams, 'itemCount' | 'itemIds'>;
  onItemsToEvaluateChange: (itemsToEvaluate: Pick<EvaluateTracesParams, 'itemCount' | 'itemIds'>) => void;
}) => {
  const [displayPickCustomTracesModal, setDisplayPickCustomTracesModal] = useState(false);
  const itemsToEvaluateDropdownValue = useMemo(() => {
    if (!isEmpty(itemsToEvaluate.itemIds)) {
      return PickerOption.CUSTOM;
    }
    return coerceToEnum(PickerOption, String(itemsToEvaluate.itemCount), PickerOption.LAST_10_TRACES);
  }, [itemsToEvaluate]);

  return (
    <>
      <DialogCombobox
        componentId="mlflow.experiment-scorers.form.traces-picker"
        id="mlflow.experiment-scorers.form.traces-picker"
        value={[itemsToEvaluateDropdownValue]}
      >
        <DialogComboboxCustomButtonTriggerWrapper>
          <Button
            componentId="mlflow.experiment-scorers.form.traces-picker.trigger"
            size="small"
            endIcon={<ChevronDownIcon />}
          >
            {itemsToEvaluateDropdownValue === PickerOption.CUSTOM ? (
              <FormattedMessage
                defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
                description="Label for the number of traces selected"
                values={{ count: itemsToEvaluate.itemIds?.length }}
              />
            ) : (
              <FormattedMessage {...PickerOptionsLabels[itemsToEvaluateDropdownValue]} />
            )}
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent align="end">
          <DialogComboboxOptionList>
            {[PickerOption.LAST_TRACE, PickerOption.LAST_5_TRACES, PickerOption.LAST_10_TRACES].map((value) => (
              <DialogComboboxOptionListSelectItem
                value={value}
                checked={itemsToEvaluateDropdownValue === value}
                onChange={() => {
                  onItemsToEvaluateChange({ itemCount: Number(value), itemIds: undefined });
                }}
              >
                <FormattedMessage {...PickerOptionsLabels[value]} />
              </DialogComboboxOptionListSelectItem>
            ))}
            <DialogComboboxOptionListSelectItem
              value={PickerOption.CUSTOM}
              checked={itemsToEvaluateDropdownValue === PickerOption.CUSTOM}
              onChange={() => {
                setDisplayPickCustomTracesModal(true);
              }}
            >
              <FormattedMessage {...PickerOptionsLabels[PickerOption.CUSTOM]} />
            </DialogComboboxOptionListSelectItem>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
      {displayPickCustomTracesModal && (
        <SelectTracesModal
          onClose={() => {
            onItemsToEvaluateChange({ ...itemsToEvaluate });
            setDisplayPickCustomTracesModal(false);
          }}
          onSuccess={(traceIds) => {
            if (!isEmpty(traceIds)) {
              onItemsToEvaluateChange({ itemCount: undefined, itemIds: traceIds });
            }
            setDisplayPickCustomTracesModal(false);
          }}
          initialTraceIdsSelected={itemsToEvaluate.itemIds}
        />
      )}
    </>
  );
};
