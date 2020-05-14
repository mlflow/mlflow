import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Button } from 'antd';

export class IconButton extends Component {
  static propTypes = {
    children: PropTypes.oneOfType([PropTypes.arrayOf(PropTypes.node), PropTypes.node]).isRequired,
    style: PropTypes.object,
    restProps: PropTypes.object,
  };

  render() {
    const { style, children, ...restProps } = this.props;
    return (
      <Button type='link' style={{ padding: 0, ...style }} {...restProps}>
        {children}
      </Button>
    );
  }
}
