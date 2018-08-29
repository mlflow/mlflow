import React from 'react';

import { Form, Select as AntdSelect } from 'antd';
import {formItemLayout} from './layout';

const FormItem = Form.Item;
const Option = AntdSelect.Option;

export default function Select(
  { field, form: { touched, errors }, label, options, ...props },
) {
  const error = errors[field.name];
  return <FormItem
    {...formItemLayout}
    label={label}
    validateStatus={touched && error ? 'error' : null}
    help={touched && error ? error : null}

  >
    <AntdSelect
      {...field}
      {...props}
      onChange={(e) => {
        // https://github.com/jaredpalmer/formik/issues/187
        e.persist = () => {};
        field.onChange(e);
      }}
      dropdownStyle={{zIndex: 2000}}
    >
      { options.map(
        ({label: optionLabel, value: optionValue}) =>
          <Option key={optionValue} value={optionValue}>
            {optionLabel}
          </Option>
      )
      }
    </AntdSelect>
  </FormItem>;
}
