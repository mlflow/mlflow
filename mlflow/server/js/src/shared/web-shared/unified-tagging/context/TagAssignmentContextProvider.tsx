import invariant from 'invariant';
import { createContext, useContext } from 'react';
import type { FieldValues, ArrayPath, FieldArray } from 'react-hook-form';

import type { UseTagAssignmentFormReturn } from '../hooks/useTagAssignmentForm';

export const TagAssignmentContext = createContext<UseTagAssignmentFormReturn | null>(null);

export function TagAssignmentContextProvider<
  T extends FieldValues = FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
>({ children, ...props }: { children: React.ReactNode } & UseTagAssignmentFormReturn<T, K, V>) {
  return <TagAssignmentContext.Provider value={props as any}>{children}</TagAssignmentContext.Provider>;
}

export function useTagAssignmentContext<
  T extends FieldValues = FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
>() {
  const context = useContext(TagAssignmentContext as React.Context<UseTagAssignmentFormReturn<T, K, V> | null>);
  invariant(context, 'useTagAssignmentContext must be used within a TagAssignmentRoot');
  return context;
}
