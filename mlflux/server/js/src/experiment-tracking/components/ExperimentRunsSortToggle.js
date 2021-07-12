import React from 'react';
import PropTypes from 'prop-types';

export default class ExperimentRunsSortToggle extends React.Component {
  static propTypes = {
    children: PropTypes.node,
    bsRole: PropTypes.string,
    bsClass: PropTypes.string,
  };

  render() {
    // eslint-disable-next-line no-unused-vars
    const { bsRole, bsClass, ...otherProps } = this.props;
    return <span {...otherProps}>{this.props.children}</span>;
  }
}
