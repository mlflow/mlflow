// Do not modify this file

import type { ControllerProps, FieldValues, Path } from 'react-hook-form';
import { Controller } from 'react-hook-form';

import { TagAssignmentInput } from './TagAssignmentField/TagAssignmentInput';
import { useTagAssignmentContext } from '../context/TagAssignmentContextProvider';

interface TagAssignmentValueProps<T extends FieldValues> {
  rules?: ControllerProps<T>['rules'];
  index: number;
  render?: ControllerProps<T>['render'];
}

export function TagAssignmentValue<T extends FieldValues>({ rules, index, render }: TagAssignmentValueProps<T>) {
  const { name, valueProperty } = useTagAssignmentContext<T>();

  return (
    <Controller
      rules={rules}
      name={`${name}.${index}.${valueProperty}` as Path<T>}
      render={({ field, fieldState, formState }) => {
        if (render) {
          return render({ field, fieldState, formState });
        }

        return (
          <TagAssignmentInput
            componentId="TagAssignmentValue.Default.Input"
            errorMessage={fieldState.error?.message}
            {...field}
          />
        );
      }}
    />
  );
}
