import React from 'react';

import MaterialSelect from '@material-ui/core/Select';
import FormControl from '@material-ui/core/FormControl';
import FormHelperText from '@material-ui/core/FormHelperText';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';

export default function Select(
  { field, form: { touched, errors }, label, className, options, ...props }
) {
  const error = errors[field.name];
  return <FormControl className={className}>
    <InputLabel htmlFor={field.name}>{label}</InputLabel>
    <MaterialSelect
      {...field}
      {...props}
    >
      { options.map(
        ({label: optionLabel, value: optionValue}) =>
          <MenuItem key={optionValue} value={optionValue}>{optionLabel}</MenuItem>
      )}
    </MaterialSelect>
    {touched[field.name] && error ?
      <FormHelperText>{error}</FormHelperText> : null
    }
  </FormControl>
}
