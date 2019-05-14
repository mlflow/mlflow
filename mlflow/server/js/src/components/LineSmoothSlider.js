import React from 'react';
import { Slider, InputNumber, Row, Col } from 'antd';
import PropTypes from 'prop-types';

export class LineSmoothSlider extends React.Component {
  static propTypes = {
    min: PropTypes.number.isRequired,
    max: PropTypes.number.isRequired,
    handleLineSmoothChange: PropTypes.func.isRequired,
  };

  state = {
    inputValue: 0,
  };

  onChange = (value) => {
    if (Number.isNaN(value)) {
      return;
    }
    this.setState({
      inputValue: value,
    });
    this.props.handleLineSmoothChange(value);
  };

  render() {
    const { min, max } = this.props;
    const { inputValue } = this.state;
    return (
      <Row>
        <Col span={12}>
          <Slider
            min={min}
            max={max}
            onChange={this.onChange}
            value={typeof inputValue === 'number' ? inputValue : 0}
            step={0.01}
          />
        </Col>
        <Col span={4}>
          <InputNumber
            min={min}
            max={max}
            style={{ marginLeft: 16 }}
            step={0.01}
            value={inputValue}
            onChange={this.onChange}
          />
        </Col>
      </Row>
    );
  }
}
