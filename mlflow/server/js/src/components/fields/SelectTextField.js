import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Select from 'react-select';

/**
 * Component resembling an HTML <select> element, but with support for selecting options by entering
 * their names into a text field.
 */
export default class SelectTextField extends Component {
  static propTypes = {
    options: PropTypes.arrayOf(String).isRequired,
  };

  render() {
    const { options, ...otherProps } = this.props;
    const formattedOptions = options.map((option) => {
      return {label: option, value: option};
    });

    return <Select
      options={formattedOptions}
      {...otherProps}
    />;
  }
}
