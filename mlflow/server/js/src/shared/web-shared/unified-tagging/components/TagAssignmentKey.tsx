// Do not modify this file

import React from 'react';
import type { ControllerProps, FieldValues, Path, UseControllerProps } from 'react-hook-form';
import { Controller } from 'react-hook-form';

import { TagAssignmentInput } from './TagAssignmentField/TagAssignmentInput';
import { useTagAssignmentContext } from '../context/TagAssignmentContextProvider';

interface TagAssignmentKeyProps<T extends FieldValues> {
  index: number;
  rules?: UseControllerProps<T>['rules'];
  render?: ControllerProps<T>['render'];
}

export function TagAssignmentKey<T extends FieldValues>({ index, rules, render }: TagAssignmentKeyProps<T>) {
  const { name, keyProperty, getTagsValues, emptyValue, appendIfPossible } = useTagAssignmentContext<T>();

  return (
    <Controller
      name={`${name}.${index}.${keyProperty}` as Path<T>}
      rules={rules}
      render={({ field, fieldState, formState }) => {
        const legacyChange = field.onChange;

        function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
          legacyChange(e);

          const tags = getTagsValues();
          if (!tags?.at(-1)?.[keyProperty]) return;
          appendIfPossible(emptyValue, { shouldFocus: false });
        }
        field.onChange = handleChange;

        if (render) {
          return render({ field, fieldState, formState });
        }

        return (
          <TagAssignmentInput
            componentId="TagAssignmentKey.Default.Input"
            errorMessage={fieldState.error?.message}
            {...field}
          />
        );
      }}
    />
  );
}
