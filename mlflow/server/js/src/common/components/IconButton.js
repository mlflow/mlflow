import React from 'react';
import PropTypes from 'prop-types';
import { Button } from 'antd';

export const IconButton = ({ icon, className, style, ...restProps }) => {
  return (
    <Button type='link' className={className} style={{ padding: 0, ...style }} {...restProps}>
      {icon}
    </Button>
  );
};

IconButton.propTypes = {
  icon: PropTypes.node.isRequired,
  style: PropTypes.object,
  className: PropTypes.string,
  restProps: PropTypes.object,
};
