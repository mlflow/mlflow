import React from "react";
import PropTypes from "prop-types";

export default class ExperimentRunsSortToggle extends React.Component {
  static propTypes = {
    children: PropTypes.arrayOf(PropTypes.element),
    className: PropTypes.string,
  };

  render() {
    return (
      <span className={this.props.className}>
        {this.props.children}
      </span>
    );
  }
}
