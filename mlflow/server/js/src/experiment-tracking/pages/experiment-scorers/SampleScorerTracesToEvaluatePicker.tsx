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
import { ScorerFormData } from './utils/scorerTransformUtils';
import { useFormContext } from 'react-hook-form';
import { MAX_SELECTED_ITEM_COUNT, ScorerEvaluationScope } from './constants';
import { SelectSessionsModal } from '../../components/SelectSessionsModal';

enum PickerOption {
  'LAST_TRACE_OR_SESSION' = '1',
  'CUSTOM' = 'custom',
}

const PickerOptionsLabelsForTraces = {
  [PickerOption.LAST_TRACE_OR_SESSION]: defineMessage({
    defaultMessage: 'Last trace',
    description: 'Option for last trace',
  }),
  [PickerOption.CUSTOM]: defineMessage({
    defaultMessage: 'Select traces',
    description: 'Option for selecting custom traces',
  }),
};

const PickerOptionsLabelsForSessions = {
  [PickerOption.LAST_TRACE_OR_SESSION]: defineMessage({
    defaultMessage: 'Last session',
    description: 'Option for last session',
  }),
  [PickerOption.CUSTOM]: defineMessage({
    defaultMessage: 'Select sessions',
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
  const { watch } = useFormContext<ScorerFormData>();

  const evaluationScope = coerceToEnum(ScorerEvaluationScope, watch('evaluationScope'), ScorerEvaluationScope.TRACES);

  const PickerOptionsLabels =
    evaluationScope === ScorerEvaluationScope.TRACES ? PickerOptionsLabelsForTraces : PickerOptionsLabelsForSessions;

  const [displayPickCustomTracesModal, setDisplayPickCustomTracesModal] = useState(false);
  const itemsToEvaluateDropdownValue = useMemo(() => {
    if (!isEmpty(itemsToEvaluate.itemIds)) {
      return PickerOption.CUSTOM;
    }
    return coerceToEnum(PickerOption, String(itemsToEvaluate.itemCount), PickerOption.LAST_TRACE_OR_SESSION);
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
              evaluationScope === ScorerEvaluationScope.TRACES ? (
                <FormattedMessage
                  defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
                  description="Label for the number of traces selected"
                  values={{ count: itemsToEvaluate.itemIds?.length }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="{count, plural, one {1 session selected} other {# sessions selected}}"
                  description="Label for the number of sessions selected"
                  values={{ count: itemsToEvaluate.itemIds?.length }}
                />
              )
            ) : (
              <FormattedMessage {...PickerOptionsLabels[itemsToEvaluateDropdownValue]} />
            )}
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent align="end">
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSelectItem
              value={PickerOption.LAST_TRACE_OR_SESSION}
              checked={itemsToEvaluateDropdownValue === PickerOption.LAST_TRACE_OR_SESSION}
              onChange={() => {
                onItemsToEvaluateChange({ itemCount: Number(PickerOption.LAST_TRACE_OR_SESSION), itemIds: undefined });
              }}
            >
              <FormattedMessage {...PickerOptionsLabels[PickerOption.LAST_TRACE_OR_SESSION]} />
            </DialogComboboxOptionListSelectItem>
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
      {displayPickCustomTracesModal && evaluationScope === ScorerEvaluationScope.TRACES && (
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
          maxTraceCount={MAX_SELECTED_ITEM_COUNT}
        />
      )}
      {displayPickCustomTracesModal && evaluationScope === ScorerEvaluationScope.SESSIONS && (
        <SelectSessionsModal
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
          initialSessionIdsSelected={itemsToEvaluate.itemIds}
          maxSessionCount={MAX_SELECTED_ITEM_COUNT}
        />
      )}
    </>
  );
};
