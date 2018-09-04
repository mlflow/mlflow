import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { FormGroup, ControlLabel, FormControl, HelpBlock } from 'react-bootstrap';

class TextField extends Component {
  static propTypes = {
    field: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
    label: PropTypes.string.isRequired,
    autoFocus: PropTypes.bool,
  }

  render() {
    const field = this.props.field;
    const { touched, errors } = this.props.form;
    const label = this.props.label;
    const handleFocus = (event) => {
      event.target.select();
    };
    const error = errors && errors[field.name];
    return (<FormGroup
      controlId={field.name}
      validationState={touched && touched[field.name] && error ? 'error' : null}
    >
      <ControlLabel>{label}</ControlLabel>
      <FormControl
        type="text"
        {...field}
        onFocus={handleFocus}
        autoFocus={this.props.autoFocus}
      />
      { touched && touched[field.name] && error ? <HelpBlock>{error}</HelpBlock> : null }
    </FormGroup>);
  }
}

export default TextField;
