import React from 'react';

import { Form, Checkbox as AntdCheckbox} from 'antd';
import {formItemLayout} from './layout';

const FormItem = Form.Item;


export default function Checkbox(
  { field, form: { touched, errors }, label, ...props },
) {
  return <FormItem
    {...formItemLayout}
    label=' '
  >
    <AntdCheckbox
      {...field}
      {...props}
      onChange={(e) => {
        // https://github.com/jaredpalmer/formik/issues/187
        e.persist = () => {};
        field.onChange(e);
      }}
      checked={field.value ? true : false}
    >
      {label}
    </AntdCheckbox>
  </FormItem>

}
