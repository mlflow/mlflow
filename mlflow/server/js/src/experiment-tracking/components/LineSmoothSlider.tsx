import React from 'react';
import { Slider, InputNumber, Row, Col } from 'antd';
import { WithDesignSystemThemeHoc, DesignSystemHocProps } from '@databricks/design-system';

type Props = {
  min?: number;
  max?: number;
  step?: number;
  handleLineSmoothChange: (...args: any[]) => any;
  defaultValue: number;
  disabled?: boolean;
} & DesignSystemHocProps;

class LineSmoothSliderImpl extends React.Component<Props> {
  onChange = (value: any) => {
    if (Number.isNaN(value)) {
      return;
    }
    this.setState({
      inputValue: value,
    });
    this.props.handleLineSmoothChange(value);
  };

  render() {
    const {
      min,
      max,
      step = 1,
      disabled,
      designSystemThemeApi: { theme },
      defaultValue,
    } = this.props;

    // Until DuBois <Slider /> is under development, let's override default antd palette
    const sliderColor = disabled ? theme.colors.actionDisabledText : theme.colors.primary;

    return (
      <Row>
        <Col span={18}>
          <Slider
            disabled={disabled}
            min={min}
            max={max}
            onChange={this.onChange}
            value={typeof defaultValue === 'number' ? defaultValue : 1}
            step={step}
            trackStyle={{ background: sliderColor }}
            handleStyle={{ background: sliderColor, borderColor: sliderColor }}
          />
        </Col>
        <Col span={2}>
          <InputNumber
            disabled={disabled}
            min={min}
            max={max}
            style={{ marginLeft: 16, width: 60 }}
            step={step}
            value={typeof defaultValue === 'number' ? defaultValue : 1}
            onChange={this.onChange}
            data-test-id='InputNumber'
          />
        </Col>
      </Row>
    );
  }
}

export const LineSmoothSlider = WithDesignSystemThemeHoc(LineSmoothSliderImpl);
