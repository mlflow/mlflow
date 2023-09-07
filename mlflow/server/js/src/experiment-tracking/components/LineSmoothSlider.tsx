/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Slider, InputNumber, Row, Col } from 'antd';

type Props = {
  min: number;
  max: number;
  handleLineSmoothChange: (...args: any[]) => any;
  defaultValue: number;
};

type State = any;

export class LineSmoothSlider extends React.Component<Props, State> {
  state = {
    inputValue: this.props.defaultValue,
  };

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
    const { min, max } = this.props;
    const { inputValue } = this.state;
    return (
      <Row>
        <Col span={12}>
          <Slider
            min={min}
            max={max}
            onChange={this.onChange}
            value={typeof inputValue === 'number' ? inputValue : 1}
            step={1}
          />
        </Col>
        <Col span={4}>
          <InputNumber
            min={min}
            max={max}
            style={{ marginLeft: 16 }}
            step={1}
            value={inputValue}
            onChange={this.onChange}
            data-test-id='InputNumber'
          />
        </Col>
      </Row>
    );
  }
}
