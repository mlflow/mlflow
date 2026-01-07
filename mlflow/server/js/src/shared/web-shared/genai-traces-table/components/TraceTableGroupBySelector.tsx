import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { TraceGroupByConfig } from '../types';

export interface TraceTableGroupBySelectorProps {
  groupByConfig: TraceGroupByConfig | null;
  setGroupByConfig: (config: TraceGroupByConfig | null) => void;
  hasSessionTraces: boolean;
}

export const TraceTableGroupBySelector = ({
  groupByConfig,
  setGroupByConfig,
  hasSessionTraces,
}: TraceTableGroupBySelectorProps) => {
  const isSessionGroupingEnabled = groupByConfig?.mode === 'session';

  const toggleSessionGrouping = () => {
    if (isSessionGroupingEnabled) {
      setGroupByConfig(null);
    } else {
      setGroupByConfig({ mode: 'session' });
    }
  };

  // Don't render if there are no session traces
  if (!hasSessionTraces) {
    return null;
  }

  return (
    <DialogCombobox
      componentId="mlflow.traces-table.group-by-selector"
      label={<FormattedMessage defaultMessage="Group by" description="Label for the trace table group by selector" />}
      multiSelect
    >
      <DialogComboboxTrigger />
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          <DialogComboboxOptionListCheckboxItem
            key="session"
            value="Session ID"
            checked={isSessionGroupingEnabled}
            onChange={toggleSessionGrouping}
          />
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
