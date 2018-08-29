import React from 'react';

import MaterialTextField from '@material-ui/core/TextField';

export default function TextField(
  { field, form: { touched, errors }, ...props },
) {
  const error = errors[field.name];
  return <MaterialTextField
    {...field}
    {...props}
    error={Boolean(touched[field.name] && error)}
    helperText={touched[field.name] && error}
  />;
}
