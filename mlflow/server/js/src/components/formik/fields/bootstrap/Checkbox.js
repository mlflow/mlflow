import React from 'react';

import { Checkbox as BootstrapCheckbox} from 'react-bootstrap';

export default function Checkbox(
  { field, form: { touched, errors }, label, ...props },
) {
  return <BootstrapCheckbox
        {...field}
        {...props}
        checked={field.value ? true : false}
  >
    {label}
  </BootstrapCheckbox>;
}
