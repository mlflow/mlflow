import React from 'react';
import { FormGroup, ControlLabel, FormControl, HelpBlock } from 'react-bootstrap';

export default function TextField({ field, form: { touched, errors }, label, ...props }) {
  const handleFocus = (event) => {
    event.target.select();
  };
  const error = errors[field.name];
  return (<FormGroup
    controlId={field.name}
    validationState={touched[field.name] && error ? 'error' : null}
  >
    <ControlLabel>{label}</ControlLabel>
    <FormControl
      type="text"
      {...field}
      onFocus={handleFocus}
      autoFocus={props && props.autoFocus}
    />
    { touched[field.name] && error ? <HelpBlock>{error}</HelpBlock> : null }
  </FormGroup>);
}
