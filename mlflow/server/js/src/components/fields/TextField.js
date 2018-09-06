import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './TextField.css';

import { FormGroup, ControlLabel, FormControl, HelpBlock } from 'react-bootstrap';

class TextField extends Component {
  static propTypes = {
    field: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
    label: PropTypes.string.isRequired,
  }

  render() {
    const { form: {touched, errors}, label, field, ...props } = this.props;
    const error = errors && errors[field.name];
    return (<FormGroup
      controlId={field.name}
      validationState={touched && touched[field.name] && error ? 'error' : null}
      className="mlflow-formgroup"
    >
      <ControlLabel>{label}</ControlLabel>
      <FormControl
        type="text"
        {...field}
        {...props}
      />
      { touched && touched[field.name] && error ? <HelpBlock>{error}</HelpBlock> : null }
    </FormGroup>);
  }
}

export default TextField;
