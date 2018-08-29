import React from 'react';

import { FormGroup, ControlLabel, Col, FormControl, HelpBlock } from 'react-bootstrap';

export default function TextField(
  { field, form: { touched, errors }, label, ...props },
) {
  const error = errors[field.name];
  return <FormGroup
    controlId={field.name}
    validationState={touched[field.name] && error ? 'error' : null}
  >
    <ControlLabel>{label}</ControlLabel>
    <FormControl
      type="text"
      {...field}
    />
    { touched[field.name] && error ? <HelpBlock>{error}</HelpBlock> : null }
  </FormGroup>
}
