import { renderHook } from '@testing-library/react';
import { useForm, FormProvider } from 'react-hook-form';
import { IntlProvider } from 'react-intl';

import { useTagAssignmentForm } from './useTagAssignmentForm';

const DefaultWrapper = ({ children }: { children: React.ReactNode }) => {
  return <IntlProvider locale="en">{children}</IntlProvider>;
};

describe('useTagAssignmentForm', () => {
  it('should use passed form as prop', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ input: string; tags: { key: string; value: undefined }[] }>({ defaultValues: { input: 'test_input' } }),
    );
    const { result } = renderHook(
      () =>
        useTagAssignmentForm({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          form: formResult.current,
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { wrapper: DefaultWrapper },
    );
    const values = result.current.form.getValues();
    expect(values.tags).toStrictEqual([{ key: '', value: undefined }]);
    expect(values.input).toBe('test_input');
  });

  it('should use context form if no form prop is passed', () => {
    const Wrapper = ({ children }: { children: React.ReactNode }) => {
      const methods = useForm<{ input: string; tags: { key: string; value: undefined }[] }>({
        defaultValues: { input: 'test_input' },
      });
      return (
        <FormProvider {...methods}>
          <IntlProvider locale="en">{children}</IntlProvider>
        </FormProvider>
      );
    };

    const { result } = renderHook(
      () =>
        useTagAssignmentForm({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { wrapper: Wrapper },
    );

    const values = result.current.form.getValues();
    expect(values['tags']).toStrictEqual([{ key: '', value: undefined }]);
    expect(values['input']).toBe('test_input');
  });

  it('should add an empty value on default values provided by form context', () => {
    const Wrapper = ({ children }: { children: React.ReactNode }) => {
      const methods = useForm<{ input: string; tags: { key: string; value: string | undefined }[] }>({
        defaultValues: { input: 'test_input', tags: [{ key: 'defaultKey', value: 'defaultValue' }] },
      });
      return (
        <FormProvider {...methods}>
          <IntlProvider locale="en">{children}</IntlProvider>
        </FormProvider>
      );
    };

    const { result } = renderHook(
      () =>
        useTagAssignmentForm({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { wrapper: Wrapper },
    );

    const values = result.current.form.getValues();
    expect(values['tags']).toStrictEqual([
      { key: 'defaultKey', value: 'defaultValue' },
      { key: '', value: undefined },
    ]);
    expect(values['input']).toBe('test_input');
  });

  it('should throw an error if no form is passed and not in a form context', () => {
    expect(() =>
      renderHook(
        () =>
          useTagAssignmentForm({
            name: 'tags',
            emptyValue: { key: '', value: undefined },
            keyProperty: 'key',
            valueProperty: 'value',
          }),
        { wrapper: DefaultWrapper },
      ),
    ).toThrow('Nest your component on a FormProvider or pass a form prop');
  });

  it('should throw an error if default values are passed and in a form context', () => {
    const Wrapper = ({ children }: { children: React.ReactNode }) => {
      const methods = useForm<{ input: string; tags: { key: string; value: string | undefined }[] }>({
        defaultValues: { input: 'test_input' },
      });
      return (
        <FormProvider {...methods}>
          <IntlProvider locale="en">{children}</IntlProvider>
        </FormProvider>
      );
    };

    expect(() =>
      renderHook(
        () =>
          useTagAssignmentForm({
            name: 'tags',
            emptyValue: { key: '', value: undefined },
            keyProperty: 'key',
            valueProperty: 'value',
            defaultValues: [{ key: 'defaultKey', value: 'defaultValue' }],
          }),
        { wrapper: Wrapper },
      ),
    ).toThrow('Define defaultValues at form context level');
  });

  it('should use empty value if no default values are passed', () => {
    const { result: formResult } = renderHook(() => useForm());
    const { result } = renderHook(
      () =>
        useTagAssignmentForm({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          keyProperty: 'key',
          valueProperty: 'value',
          form: formResult.current,
        }),
      { wrapper: DefaultWrapper },
    );

    const values = result.current.form.getValues();
    expect(values['tags']).toStrictEqual([{ key: '', value: undefined }]);
  });

  it('should use default values + empty value if default values are passed', () => {
    const { result: formResult } = renderHook(() => useForm<{ tags: { key: string; value: string | undefined }[] }>());
    const defaultValues = [{ key: 'defaultKey', value: 'defaultValue' }];
    const { result } = renderHook(
      () =>
        useTagAssignmentForm<{ tags: { key: string; value: string | undefined }[] }>({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          defaultValues,
          form: formResult.current,
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { wrapper: DefaultWrapper },
    );

    const values = result.current.form.getValues();
    expect(values.tags).toStrictEqual([
      { key: 'defaultKey', value: 'defaultValue' },
      { key: '', value: undefined },
    ]);
  });

  it('should not add the empty value to the form if maxLength is reached', () => {
    const { result: formResult } = renderHook(() => useForm<{ tags: { key: string; value: string | undefined }[] }>());
    const defaultValues = [
      { key: 'key1', value: 'value1' },
      { key: 'key2', value: 'value2' },
    ];
    const { result } = renderHook(
      () =>
        useTagAssignmentForm<{ tags: { key: string; value: string | undefined }[] }>({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          defaultValues,
          maxLength: 2,
          form: formResult.current,
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { wrapper: DefaultWrapper },
    );

    const values = result.current.form.getValues();
    expect(values.tags).toStrictEqual([
      { key: 'key1', value: 'value1' },
      { key: 'key2', value: 'value2' },
    ]);
  });

  it('should wait for loading before resetting', () => {
    const { result: formResult } = renderHook(() => useForm());
    const { result, rerender } = renderHook(
      ({ loading }) =>
        useTagAssignmentForm({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          loading,
          form: formResult.current,
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { initialProps: { loading: true }, wrapper: DefaultWrapper },
    );

    const initialValues = result.current.form.getValues();
    expect(initialValues['tags']).toStrictEqual([]);

    rerender({ loading: false });

    const values = result.current.form.getValues();
    expect(values['tags']).toStrictEqual([{ key: '', value: undefined }]);
  });

  it('should not override the other filled values when setting default values', () => {
    const { result: formResult } = renderHook(() =>
      useForm<{ tags: { key: string; value: string | undefined }[]; input1: string; input2: string }>({
        defaultValues: {
          input1: 'test_input1',
          input2: 'test_input2',
        },
      }),
    );
    const { result } = renderHook(
      () =>
        useTagAssignmentForm({
          name: 'tags',
          emptyValue: { key: '', value: undefined },
          form: formResult.current,
          keyProperty: 'key',
          valueProperty: 'value',
        }),
      { wrapper: DefaultWrapper },
    );

    const values = result.current.form.getValues();
    expect(values.tags).toStrictEqual([{ key: '', value: undefined }]);
    expect(values.input1).toBe('test_input1');
    expect(values.input2).toBe('test_input2');
  });
});
