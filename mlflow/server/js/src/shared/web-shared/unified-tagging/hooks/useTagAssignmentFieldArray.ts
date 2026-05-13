import invariant from 'invariant';
import { useCallback, useState } from 'react';
import type { ArrayPath, FieldArray, FieldArrayMethodProps, FieldValues, Path } from 'react-hook-form';
import { useFieldArray, useFormContext } from 'react-hook-form';

import { useIntl } from 'react-intl';
import type { IntlShape } from 'react-intl';

import type { UseTagAssignmentProps } from './useTagAssignmentForm';

function getTagAssignmentRules(maxLength: number | undefined, intl: IntlShape) {
  if (maxLength === undefined) return undefined;
  if (maxLength === 0) {
    invariant(false, 'maxLength must be greater than 0');
  }
  return {
    maxLength: {
      value: maxLength,
      message: intl.formatMessage(
        {
          defaultMessage: `You can set a maximum of {maxLength} values`,
          description:
            'Error message when trying to submit a key-value pair form with more than the maximum allowed values',
        },
        {
          maxLength,
        },
      ),
    },
  };
}

type UseTagAssignmentFieldArrayProps<
  T extends FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
> = Pick<UseTagAssignmentProps<T, K, V>, 'name' | 'maxLength' | 'emptyValue' | 'form' | 'keyProperty'>;

/**
 * Alternative to useTagAssignmentForm that only provides a wrapper around RHF's useFieldArray without any
 * side effects to initialize the form state.
 *
 * As with useFieldArray, the caller is expected to manage the form state themselves using these methods.
 * For conformance to the unified tagging pattern, there are 2 key things you are responsible for:
 *   1. Initialize the form state with an empty tag
 *   2. Call appendIfPossible when the user inputs something into the last tag key field
 */
export function useTagAssignmentFieldArray<
  T extends FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
>({ name, maxLength, emptyValue, form, keyProperty }: UseTagAssignmentFieldArrayProps<T, K, V>) {
  const intl = useIntl();

  const formCtx = useFormContext<T>();
  const shouldUseFormContext = Boolean(formCtx) && !form;
  const internalForm = shouldUseFormContext ? formCtx : form;

  invariant(internalForm, 'Nest your component on a FormProvider or pass a form prop');

  const [_emptyValue] = useState(emptyValue);
  const {
    append: originalAppend,
    update,
    remove: originalRemove,
    ...fieldArrayMethods
  } = useFieldArray<T, K>({
    name,
    control: internalForm.control,
    rules: getTagAssignmentRules(maxLength, intl),
  });

  const { getValues } = internalForm;

  const getTagsValues = useCallback(() => {
    return getValues(name as Path<T>) as V[] | undefined;
  }, [getValues, name]);

  const appendIfPossible = useCallback(
    (value: V | V[], options: FieldArrayMethodProps) => {
      const tags = getTagsValues();
      if (maxLength && tags && tags.length >= maxLength) return;
      originalAppend(value, options);
    },
    [getTagsValues, maxLength, originalAppend],
  );

  const removeOrUpdate = useCallback(
    (index: number) => {
      const tags = getTagsValues();
      if (tags && index === tags.length - 1) {
        return update(index, _emptyValue);
      }
      const lastTag = tags?.at(-1);
      if (lastTag?.[keyProperty]) {
        originalRemove(index);
        originalAppend(_emptyValue, { shouldFocus: false });
        return;
      }
      originalRemove(index);
    },
    [_emptyValue, getTagsValues, keyProperty, originalAppend, originalRemove, update],
  );

  return {
    form: internalForm,
    ...fieldArrayMethods,
    originalAppend,
    update,
    originalRemove,
    appendIfPossible,
    removeOrUpdate,
    getTagsValues,
  };
}

export type UseTagAssignmentFieldArrayReturn<
  T extends FieldValues = FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
> = ReturnType<typeof useTagAssignmentFieldArray<T, K, V>>;
