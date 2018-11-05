import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Select from 'react-select';

export default class SelectizeTextField extends Component {
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
      styles={{
        container: (provided) => {
          return { ...provided, zIndex: 4 };
        }
      }}
    />;
  }
}

