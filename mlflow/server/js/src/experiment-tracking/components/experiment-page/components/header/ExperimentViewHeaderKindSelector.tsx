import { ChevronDownIcon, DropdownMenu, Popover, Spinner, Tag, Tooltip } from '@databricks/design-system';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { entries } from 'lodash';
import { useMemo, useState } from 'react';
import { defineMessage, FormattedMessage } from 'react-intl';
import { ExperimentKind } from '../../../../constants';
import {
  ExperimentKindDropdownLabels,
  getSelectableExperimentKinds,
  isEditableExperimentKind,
  normalizeInferredExperimentKind,
} from '../../../../utils/ExperimentKindUtils';
import { ExperimentViewInferredKindPopover } from './ExperimentViewInferredKindPopover';

const getVisibleLabel = (kind: ExperimentKind, readOnly: boolean) => {
  if (kind === ExperimentKind.NO_INFERRED_TYPE || kind === ExperimentKind.EMPTY) {
    if (readOnly) {
      // if the user does not have permission to edit the experiment kind, we show the "None" label
      return ExperimentKindDropdownLabels[ExperimentKind.NO_INFERRED_TYPE];
    }
    return defineMessage({
      defaultMessage: 'Select a type',
      description: 'Label for the experiment type selector in the experiment view header',
    });
  }
  return ExperimentKindDropdownLabels[kind];
};

export const ExperimentViewHeaderKindSelector = ({
  value,
  inferredExperimentKind,
  onChange,
  isUpdating,
  readOnly = false,
}: {
  value?: ExperimentKind;
  inferredExperimentKind?: ExperimentKind;
  onChange?: (kind: ExperimentKind) => void;
  isUpdating?: boolean;
  readOnly?: boolean;
}) => {
  const dropdownItems = useMemo(
    () =>
      entries(ExperimentKindDropdownLabels).filter(([key]) =>
        getSelectableExperimentKinds().includes(key as ExperimentKind),
      ),
    [],
  );

  const currentValue = useMemo(() => {
    if (inferredExperimentKind) {
      return normalizeInferredExperimentKind(inferredExperimentKind);
    }
    return coerceToEnum(ExperimentKind, value, ExperimentKind.NO_INFERRED_TYPE);
  }, [value, inferredExperimentKind]);

  const visibleLabel = getVisibleLabel(currentValue, readOnly);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [displayInferencePopover, setDisplayInferencePopover] = useState(
    Boolean(inferredExperimentKind && !readOnly && isEditableExperimentKind(inferredExperimentKind)),
  );

  // Determines if we should render a dropdown or just a tag.
  const usingDropdown = isEditableExperimentKind(currentValue) && !readOnly;

  const tagElement = (
    <Tag
      icon={isUpdating ? <Spinner size="small" /> : null}
      componentId="mlflow.experiment_view.header.experiment_kind_selector"
      css={{ marginRight: 0 }}
      // Empty callback so <Tag /> renders its "clickable" UI style
      onClick={!usingDropdown ? undefined : () => {}}
    >
      {visibleLabel && <FormattedMessage {...visibleLabel} />} {usingDropdown && <ChevronDownIcon />}
    </Tag>
  );

  const tagElementWithTooltip = <ExperimentTypeTooltip>{tagElement}</ExperimentTypeTooltip>;

  if (readOnly) {
    return tagElementWithTooltip;
  }

  const dropdownElement = (
    <DropdownMenu.Root
      modal={false}
      open={dropdownOpen}
      onOpenChange={(open) => {
        setDisplayInferencePopover(false);
        setDropdownOpen(open);
      }}
    >
      {/* Mixing dropdown with tooltip requires different ordering */}
      <ExperimentTypeTooltip>
        <DropdownMenu.Trigger asChild>{tagElement}</DropdownMenu.Trigger>
      </ExperimentTypeTooltip>
      <DropdownMenu.Content align="start">
        <DropdownMenu.Arrow />
        <DropdownMenu.Label>
          <FormattedMessage
            defaultMessage="Experiment type"
            description="Label for the experiment type selector in the experiment view header"
          />
        </DropdownMenu.Label>
        {dropdownItems.map(([key, label]) => {
          const isSelected = key === currentValue;
          return (
            <DropdownMenu.CheckboxItem
              key={key}
              componentId={`mlflow.experiment_view.header.experiment_kind_selector.${key}`}
              onClick={() => onChange?.(key as ExperimentKind)}
              checked={isSelected}
            >
              <DropdownMenu.ItemIndicator />
              <FormattedMessage {...label} />
            </DropdownMenu.CheckboxItem>
          );
        })}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );

  if (displayInferencePopover && inferredExperimentKind) {
    return (
      <ExperimentViewInferredKindPopover
        inferredExperimentKind={inferredExperimentKind}
        onConfirm={async () => {
          if (inferredExperimentKind) {
            onChange?.(normalizeInferredExperimentKind(inferredExperimentKind));
          }
          setDisplayInferencePopover(false);
        }}
        onDismiss={() => setDisplayInferencePopover(false)}
        isInferredKindEditable={isEditableExperimentKind(currentValue)}
      >
        {usingDropdown ? dropdownElement : tagElementWithTooltip}
      </ExperimentViewInferredKindPopover>
    );
  }

  return usingDropdown ? dropdownElement : tagElementWithTooltip;
};

const ExperimentTypeTooltip = ({ children }: { children: React.ReactNode }) => (
  <Tooltip
    componentId="mlflow.experiment_view.header.experiment_kind_selector.tooltip"
    content={
      <FormattedMessage
        defaultMessage="Experiment type"
        description="Label for the experiment type selector in the experiment view header"
      />
    }
  >
    {children}
  </Tooltip>
);
