import React from 'react';

import { FormGroup, ControlLabel, FormControl } from 'react-bootstrap';

export default function Select(
  { field, form: { touched, errors }, label, options,  ...props },

) {
  return  <FormGroup controlId={field.name}>
    <ControlLabel>{label}</ControlLabel>
    <FormControl componentClass="select" {...field}>
      { options.map(
        ({label: optionLabel, value: optionValue}) =>
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
      )}
    </FormControl>
  </FormGroup>;
}
