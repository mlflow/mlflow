import React from "react";
import PropTypes from "prop-types";

export default class ExperimentRunsSortToggle extends React.Component {
  static propTypes = {
    children: PropTypes.arrayOf(PropTypes.element),
  };

  render() {
    return (
      <span {...this.props}>
        {this.props.children}
      </span>
    );
  }
}
