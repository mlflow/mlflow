import { SegmentedControlButton, SegmentedControlGroup } from '@databricks/design-system';

import type { InputPassFail } from '../types';

export interface LabelSchemaInputPassFailProps {
  /** The schema's `pass_fail` input definition (labels for each side). */
  input: InputPassFail;
  /**
   * Controlled selection. `true` selects the positive button, `false` the
   * negative button, `null`/`undefined` means nothing is selected yet.
   */
  value: boolean | null | undefined;
  onChange: (value: boolean) => void;
  disabled?: boolean;
  /**
   * Stable, PII-free identifier used by the Design System for telemetry.
   * Callers should namespace with their feature path (e.g.,
   * `experiment.label-schemas.review.correctness`).
   */
  componentId: string;
}

/**
 * Pass/Fail labeling widget. Renders as a two-button segmented control
 * with the schema-defined `positive_label` and `negative_label`.
 *
 * The stored assessment value is a `bool`: `true` corresponds to the
 * positive side, `false` to the negative side, mirroring the
 * `mlflow.genai.label_schemas.InputPassFail` contract.
 */
export const LabelSchemaInputPassFail = ({
  input,
  value,
  onChange,
  disabled,
  componentId,
}: LabelSchemaInputPassFailProps) => {
  return (
    <SegmentedControlGroup
      componentId={componentId}
      name={componentId}
      value={value ?? undefined}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
    >
      <SegmentedControlButton value>{input.positive_label}</SegmentedControlButton>
      <SegmentedControlButton value={false}>{input.negative_label}</SegmentedControlButton>
    </SegmentedControlGroup>
  );
};
