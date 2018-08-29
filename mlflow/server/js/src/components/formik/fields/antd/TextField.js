import React from 'react';

import { Form, Input } from 'antd';
import {formItemLayout} from './layout';

const FormItem = Form.Item

export default function TextField(
  { field, form: { touched, errors }, label, ...props },
) {
  const error = errors[field.name];
  return <FormItem
    {...formItemLayout}
    label={label}
    validateStatus={touched[field.name] && error ? 'error' : null}
    help={touched[field.name] && error ? error : null}
  >
    <Input
      type="text"
      {...field}
      {...props}
    />
  </FormItem>
}
