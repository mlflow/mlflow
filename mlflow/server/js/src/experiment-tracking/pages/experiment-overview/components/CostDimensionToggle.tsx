import React from 'react';
import { SegmentedControlGroup, SegmentedControlButton } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { CostDimension } from '../hooks/useTraceCostDimension';

interface CostDimensionToggleProps {
  componentId: string;
  value: CostDimension;
  onChange: (dimension: CostDimension) => void;
}

export const CostDimensionToggle: React.FC<CostDimensionToggleProps> = ({ componentId, value, onChange }) => (
  <SegmentedControlGroup
    name={`${componentId}-dimension`}
    componentId={componentId}
    value={value}
    onChange={({ target: { value: v } }) => onChange(v as CostDimension)}
  >
    <SegmentedControlButton value="model">
      <FormattedMessage defaultMessage="Model" description="Dimension toggle option for model" />
    </SegmentedControlButton>
    <SegmentedControlButton value="provider">
      <FormattedMessage defaultMessage="Provider" description="Dimension toggle option for provider" />
    </SegmentedControlButton>
  </SegmentedControlGroup>
);
