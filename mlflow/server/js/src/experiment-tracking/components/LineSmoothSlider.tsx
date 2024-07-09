import { Col, Row, useDesignSystemTheme } from '@databricks/design-system';
import { InputNumber, Slider } from 'antd';

interface LineSmoothSliderProps {
  max?: number;
  min?: number;
  step?: number | null;
  marks?: Record<number, string>;
  defaultValue: number | undefined;
  onChange: (value: number) => void;
  disabled?: boolean;
  onAfterChange?: (value: number) => void;
}

export const LineSmoothSlider = ({
  max,
  min,
  step,
  marks,
  defaultValue,
  onChange,
  disabled,
  onAfterChange,
}: LineSmoothSliderProps) => {
  const { theme } = useDesignSystemTheme();
  const STEP_MARKS_DISPLAY_THRESHOLD = 10;
  const INPUT_NUMBER_WIDTH = 60;
  // Until DuBois <Slider /> is under development, let's override default antd palette
  const sliderColor = disabled ? theme.colors.actionDisabledText : theme.colors.primary;

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'nowrap',
        height: '32px',
        gap: theme.spacing.md,
        paddingLeft: theme.spacing.xs, // Prevent slider from being cut off
      }}
    >
      <Slider
        css={{
          '& .ant-slider-dot': {
            display: marks && Object.keys(marks).length > STEP_MARKS_DISPLAY_THRESHOLD ? 'none' : 'inherit',
          },
          width: `calc(100% - ${INPUT_NUMBER_WIDTH + theme.spacing.md}px)`,
        }}
        disabled={disabled}
        min={min}
        max={max}
        onChange={onChange}
        value={typeof defaultValue === 'number' ? defaultValue : 1}
        trackStyle={{ background: sliderColor }}
        handleStyle={{ background: sliderColor, borderColor: sliderColor }}
        marks={marks}
        step={step}
        onAfterChange={onAfterChange}
        data-test-id="Slider"
      />
      <InputNumber
        disabled={disabled}
        min={min}
        max={max}
        css={{ width: INPUT_NUMBER_WIDTH }}
        step={step === null ? undefined : step}
        value={typeof defaultValue === 'number' ? defaultValue : 1}
        onChange={onAfterChange ? onAfterChange : onChange}
        data-test-id="InputNumber"
      />
    </div>
  );
};
