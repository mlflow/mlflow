import { renderHook } from '@testing-library/react';
import { useForm, FormProvider } from 'react-hook-form';
import { IntlProvider } from 'react-intl';

import { useTagAssignmentFieldArray } from './useTagAssignmentFieldArray';

const DefaultWrapper = ({ children }: { children: React.ReactNode }) => {
  return <IntlProvider locale="en">{children}</IntlProvider>;
};

describe('useTagAssignmentFieldArray', () => {
  it('should use passed form as prop', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ input: string; tags: { key: string; value: string }[] }>({
        defaultValues: { input: 'test_input', tags: [{ key: 'key1', value: 'value1' }] },
      }),
    );
    const { result } = renderHook(
      () =>
        useTagAssignmentFieldArray({
          name: 'tags',
          emptyValue: { key: '', value: '' },
          form: formResult.current,
          keyProperty: 'key',
        }),
      { wrapper: DefaultWrapper },
    );
    result.current.appendIfPossible({ key: 'foo', value: 'bar' }, {});

    const values = result.current.form.getValues();
    expect(values.tags).toStrictEqual([
      { key: 'key1', value: 'value1' },
      { key: 'foo', value: 'bar' },
    ]);
    expect(values.input).toBe('test_input');
  });

  it('should use context form if no form prop is passed', () => {
    const Wrapper = ({ children }: { children: React.ReactNode }) => {
      const methods = useForm<{ input: string; tags: { key: string; value: string }[] }>({
        defaultValues: { input: 'test_input', tags: [{ key: 'key1', value: 'value1' }] },
      });
      return (
        <IntlProvider locale="en">
          c<FormProvider {...methods}>{children}</FormProvider>
        </IntlProvider>
      );
    };

    const { result } = renderHook(
      () =>
        useTagAssignmentFieldArray({
          name: 'tags',
          emptyValue: { key: '', value: '' },
          keyProperty: 'key',
        }),
      { wrapper: Wrapper },
    );
    result.current.appendIfPossible({ key: 'foo', value: 'bar' }, {});

    const values = result.current.form.getValues();
    expect(values['tags']).toStrictEqual([
      { key: 'key1', value: 'value1' },
      { key: 'foo', value: 'bar' },
    ]);
    expect(values['input']).toBe('test_input');
  });

  it('should throw an error if no form is passed and not in a form context', () => {
    expect(() =>
      renderHook(
        () =>
          useTagAssignmentFieldArray({
            name: 'tags',
            emptyValue: { key: '', value: undefined },
            keyProperty: 'key',
          }),
        { wrapper: DefaultWrapper },
      ),
    ).toThrow('Nest your component on a FormProvider or pass a form prop');
  });

  it('should not add the empty value to the form via appendIfPossible if maxLength is reached', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ tags: { key: string; value: string }[] }>({
        defaultValues: {
          tags: [
            { key: 'key1', value: 'value1' },
            { key: 'key2', value: 'value2' },
          ],
        },
      }),
    );

    const { result } = renderHook(
      () =>
        useTagAssignmentFieldArray<{ tags: { key: string; value: string }[] }>({
          name: 'tags',
          emptyValue: { key: '', value: '' },
          maxLength: 2,
          form: formResult.current,
          keyProperty: 'key',
        }),
      { wrapper: DefaultWrapper },
    );

    result.current.appendIfPossible({ key: 'not-added', value: 'not-added' }, {});
    expect(result.current.getTagsValues()).toStrictEqual([
      { key: 'key1', value: 'value1' },
      { key: 'key2', value: 'value2' },
    ]);
  });

  it('should remove tag when removeOrUpdate is called for tag not at the end of the array', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ tags: { key: string; value: string }[] }>({
        defaultValues: {
          tags: [
            { key: 'key1', value: 'value1' },
            { key: 'key2', value: 'value2' },
            { key: '', value: '' },
          ],
        },
      }),
    );
    const { result } = renderHook(
      () =>
        useTagAssignmentFieldArray({
          name: 'tags',
          emptyValue: { key: '', value: '' },
          maxLength: 5,
          form: formResult.current,
          keyProperty: 'key',
        }),
      { wrapper: DefaultWrapper },
    );

    result.current.removeOrUpdate(0);

    expect(result.current.getTagsValues()).toStrictEqual([
      { key: 'key2', value: 'value2' },
      { key: '', value: '' },
    ]);
  });

  it('should set last tag to the empty value when removeOrUpdate is called for last tag', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ tags: { key: string; value: string }[] }>({
        defaultValues: {
          tags: [
            { key: 'key1', value: 'value1' },
            { key: 'key2', value: 'value2' },
          ],
        },
      }),
    );
    const { result } = renderHook(
      () =>
        useTagAssignmentFieldArray({
          name: 'tags',
          emptyValue: { key: '', value: '' },
          maxLength: 2,
          form: formResult.current,
          keyProperty: 'key',
        }),
      { wrapper: DefaultWrapper },
    );

    result.current.removeOrUpdate(1);

    expect(result.current.getTagsValues()).toStrictEqual([
      { key: 'key1', value: 'value1' },
      { key: '', value: '' },
    ]);
  });

  it('should add an empty tag to the end of the array when removeOrUpdate is called when the max number of tags are present', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ tags: { key: string; value: string }[] }>({
        defaultValues: {
          tags: [
            { key: 'key1', value: 'value1' },
            { key: 'key2', value: 'value2' },
          ],
        },
      }),
    );
    const { result } = renderHook(
      () =>
        useTagAssignmentFieldArray({
          name: 'tags',
          emptyValue: { key: '', value: '' },
          maxLength: 2,
          form: formResult.current,
          keyProperty: 'key',
        }),
      { wrapper: DefaultWrapper },
    );

    result.current.removeOrUpdate(0);

    expect(result.current.getTagsValues()).toStrictEqual([
      { key: 'key2', value: 'value2' },
      { key: '', value: '' },
    ]);
  });
});
