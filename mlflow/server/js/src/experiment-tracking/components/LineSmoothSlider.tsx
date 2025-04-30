import { useDesignSystemTheme } from '@databricks/design-system';
import { Input, Slider } from '@databricks/design-system';
import { clamp, isEmpty, isUndefined, keys } from 'lodash';
import { useState } from 'react';

const TRACK_SIZE = 20;
const THUMB_SIZE = 14;
const MARK_SIZE = 8;
const MARK_OFFSET_X = (THUMB_SIZE - MARK_SIZE) / 2;
const MARK_OFFSET_Y = (TRACK_SIZE - MARK_SIZE) / 2;

const ZINDEX_MARK = 1;
const ZINDEX_THUMB = 2;

const STEP_MARKS_DISPLAY_THRESHOLD = 10;

interface LineSmoothSliderProps {
  max?: number;
  min?: number;
  step?: number;
  marks?: Record<number, any>;
  value: number | undefined;
  onChange: (value: number) => void;
  disabled?: boolean;
  componentId?: string;
  onAfterChange?: (value: number) => void;
  className?: string;
}

// Internal helper function: finds the closest value to the given value from the marks
const getClosestValue = (marks: Record<number, string>, value: number, defaultValue: number) =>
  keys(marks).reduce(
    (prev, curr) => (Math.abs(Number(curr) - value) < Math.abs(prev - value) ? Number(curr) : Number(prev)),
    defaultValue,
  );

// Internal helper function: finds the next value based on direction (down or up) from the marks
const getNextValue = (marks: Record<number, string>, currentValue: number, direction: 'down' | 'up') =>
  direction === 'down'
    ? Math.max(
        ...Object.keys(marks)
          .filter((mark) => Number(mark) < currentValue)
          .map(Number),
      )
    : Math.min(
        ...Object.keys(marks)
          .filter((mark) => Number(mark) > currentValue)
          .map(Number),
      );

export const LineSmoothSlider = ({
  max = 1,
  min = 0,
  step,
  marks,
  value,
  onChange,
  disabled,
  onAfterChange,
  componentId,
  className,
}: LineSmoothSliderProps) => {
  const { theme } = useDesignSystemTheme();
  const shouldUseMarks = !isEmpty(marks);
  const shouldDisplayMarks = shouldUseMarks && Object.keys(marks).length < STEP_MARKS_DISPLAY_THRESHOLD;

  // Temporary value is used to store the value of the input field before it is committed
  const [temporaryValue, setTemporaryValue] = useState<number | undefined>(undefined);

  return (
    <div
      css={{
        display: 'flex',
        height: theme.general.heightSm,
        gap: theme.spacing.md,
        alignItems: 'center',
      }}
    >
      <Slider.Root
        disabled={disabled}
        css={{
          flex: 1,
          position: 'relative',
          'span:last-child': { zIndex: ZINDEX_THUMB },
        }}
        className={className}
        min={min}
        max={max}
        value={[value ?? 0]}
        onValueCommit={([newValue]) => onAfterChange?.(newValue)}
        onKeyDown={(e) => {
          // If marks are used, we want to find the next value based on direction (arrow left/down or arrow right/up)
          if (shouldUseMarks) {
            e.preventDefault();

            if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
              const nextValue = getNextValue(
                marks,
                value ?? 0,
                e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? 'down' : 'up',
              );

              onAfterChange?.(nextValue);
              onChange(nextValue);
            }
          }
        }}
        onValueChange={([newValue]) => {
          if (shouldUseMarks) {
            onChange(getClosestValue(marks, newValue, value ?? 0));
            return;
          }
          onChange(newValue);
        }}
        step={step ?? undefined}
      >
        {/* Render marks if needed */}
        {shouldDisplayMarks && (
          <div css={{ position: 'absolute', inset: 0, marginRight: THUMB_SIZE }}>
            {keys(marks).map((markPosition) => (
              <div
                key={markPosition}
                css={{
                  position: 'absolute',
                  zIndex: ZINDEX_MARK,
                  top: 0,
                  right: 0,
                  bottom: 0,
                  marginLeft: -MARK_OFFSET_X / 2,
                  marginTop: MARK_OFFSET_Y,
                  pointerEvents: 'none',
                  borderRadius: '100%',
                  backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                  height: MARK_SIZE,
                  width: MARK_SIZE,
                  opacity: 0.5,
                }}
                style={{
                  left: `${(Number(markPosition) / (max - min)) * 100}%`,
                }}
              />
            ))}
          </div>
        )}
        <Slider.Track className="TRACK">
          <Slider.Range />
        </Slider.Track>
        <Slider.Thumb css={{ position: 'relative', height: THUMB_SIZE, width: THUMB_SIZE }} />
      </Slider.Root>
      <Input
        componentId={componentId ?? 'mlflow.experiment_tracking.common.line_smooth_slider'}
        type="number"
        disabled={disabled}
        min={min}
        max={max}
        css={{ width: 'min-content' }}
        step={step}
        value={temporaryValue ?? value}
        onBlur={() => {
          // If temporary value is set, we want to commit it to the value
          if (!isUndefined(temporaryValue)) {
            if (shouldUseMarks) {
              onAfterChange?.(getClosestValue(marks, temporaryValue, value ?? 0));
              onChange(getClosestValue(marks, temporaryValue, value ?? 0));
            } else {
              onAfterChange?.(clamp(temporaryValue, min, max));
              onChange(clamp(temporaryValue, min, max));
            }
            setTemporaryValue(undefined);
          }
        }}
        onChange={({ target, nativeEvent }) => {
          // If the input event is an input event, we want to set the temporary value
          // to be commited on blur instead of directly setting the value
          if (nativeEvent instanceof InputEvent) {
            setTemporaryValue(Number(target.value));
            return;
          }

          // If using marks, find the next value based on the direction of the change
          if (shouldUseMarks) {
            const nextValue = getNextValue(marks, value ?? 0, Number(target.value) < Number(value) ? 'down' : 'up');

            onChange(nextValue);
            return;
          }

          // If not using marks, clamp the value to the min and max
          onChange(clamp(Number(target.value), min, max));
        }}
      />
    </div>
  );
};
