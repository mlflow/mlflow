import React from 'react';

import MaterialCheckbox from '@material-ui/core/Checkbox';
import FormControlLabel from '@material-ui/core/FormControlLabel';

export default function Checkbox(
  { field, form: { touched, errors }, label, ...props },
) {
  return <FormControlLabel
    control={
      <MaterialCheckbox
        {...field}

        checked={field.value ? true : false}
      />
    }
    label={label}
    {...props}
  />
}
