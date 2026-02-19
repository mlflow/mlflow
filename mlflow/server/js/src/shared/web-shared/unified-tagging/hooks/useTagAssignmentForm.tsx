import invariant from 'invariant';
import { useEffect, useState } from 'react';
import type { ArrayPath, FieldArray, FieldValues, Path, PathValue, UseFormReturn } from 'react-hook-form';
import { useFormContext } from 'react-hook-form';

import { useTagAssignmentFieldArray } from './useTagAssignmentFieldArray';

export interface UseTagAssignmentProps<
  T extends FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
> {
  name: K;
  maxLength?: number;
  emptyValue: V;
  loading?: boolean;
  defaultValues?: V[];
  form?: UseFormReturn<T>;
  keyProperty: keyof V extends string ? keyof V : never;
  valueProperty: keyof V extends string ? keyof V : never;
}

export function useTagAssignmentForm<
  T extends FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
>({
  name,
  maxLength,
  emptyValue,
  defaultValues,
  loading,
  form,
  keyProperty,
  valueProperty,
}: UseTagAssignmentProps<T, K, V>) {
  const [_emptyValue] = useState(emptyValue);

  const formCtx = useFormContext<T>();
  const shouldUseFormContext = Boolean(formCtx) && !form;
  const internalForm = shouldUseFormContext ? formCtx : form;

  invariant(internalForm, 'Nest your component on a FormProvider or pass a form prop');
  invariant(!(defaultValues && shouldUseFormContext), 'Define defaultValues at form context level');

  const { setValue } = internalForm;

  const fieldArrayMethods = useTagAssignmentFieldArray({
    name,
    maxLength,
    emptyValue,
    form: internalForm,
    keyProperty,
  });
  const getTagsValues = fieldArrayMethods.getTagsValues;

  useEffect(() => {
    if (loading) return;
    if (defaultValues) {
      const newValues = [...defaultValues];
      if (!maxLength || (maxLength && newValues.length < maxLength)) {
        newValues.push(_emptyValue);
      }
      setValue(name as Path<T>, newValues as PathValue<T, Path<T>>);
      return;
    }

    if (shouldUseFormContext) {
      const existentValues = getTagsValues() ?? [];
      if (!maxLength || (maxLength && existentValues.length < maxLength)) {
        existentValues.push(_emptyValue);
      }
      setValue(name as Path<T>, existentValues as PathValue<T, Path<T>>);
      return;
    }

    setValue(name as Path<T>, [_emptyValue] as PathValue<T, Path<T>>);
  }, [defaultValues, setValue, loading, maxLength, name, _emptyValue, shouldUseFormContext, getTagsValues]);

  return {
    ...fieldArrayMethods,
    form: internalForm,
    maxLength,
    emptyValue,
    name,
    keyProperty,
    valueProperty,
  };
}

export type UseTagAssignmentFormReturn<
  T extends FieldValues = FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
> = ReturnType<typeof useTagAssignmentForm<T, K, V>>;
