import React from "react";
import PropTypes from 'prop-types';

export class Checkbox extends React.Component {
    static propTypes = {
      label: PropTypes.string,
      isSelected: PropTypes.bool,
      onCheckboxChange: PropTypes.func,
    };

    render() {
      const {
        label,
        isSelected,
        onCheckboxChange,
      } = this.props;
      return (
        <div className="form-check">
        <label>
          <input
            type="checkbox"
            name={label}
            defaultChecked={isSelected}
            onChange={onCheckboxChange}
            className="form-check-input"
          />
          {label}
        </label>
      </div>
      );
    }
  }