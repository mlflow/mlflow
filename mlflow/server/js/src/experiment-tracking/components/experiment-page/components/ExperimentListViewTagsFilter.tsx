import { useFieldArray, useForm } from 'react-hook-form';
import { OPERATORS, TagFilter } from '../hooks/useTagsFilter';
import { Button, CloseIcon, PlusIcon, RHFControlledComponents, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { Interpolation } from '@emotion/react';
import { Theme } from '@databricks/design-system/dist/theme';
import { Fragment } from 'react';

const EMPTY_TAG = { key: '', value: '', operator: 'IS' } as const;

type Props = {
  tagsFilter: TagFilter[];
  onSave: (_: TagFilter[]) => void;
};

export function ExperimentListViewTagsFilter({ tagsFilter, onSave }: Props) {
  const { control, handleSubmit } = useForm<{ tagsFilter: TagFilter[] }>({
    defaultValues: { tagsFilter: tagsFilter.length === 0 ? [EMPTY_TAG] : tagsFilter },
  });
  const { fields, append, remove } = useFieldArray({ control, name: 'tagsFilter' });
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const labelStyles: Interpolation<Theme> = {
    fontWeight: theme.typography.typographyBoldFontWeight,
  };

  return (
    <form
      onSubmit={handleSubmit((data) => onSave(data.tagsFilter))}
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, padding: theme.spacing.md }}
    >
      <fieldset
        css={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, auto)',
          gap: theme.spacing.sm,
        }}
      >
        <label htmlFor={`${fields[0].id}-key`} css={labelStyles}>
          Key
        </label>
        <label htmlFor={`${fields[0].id}-op`} css={labelStyles}>
          Opereator
        </label>
        <label htmlFor={`${fields[0].id}-value`} css={labelStyles}>
          Value
        </label>
        <label></label>
        {fields.map((field, index) => (
          <Fragment key={field.id}>
            <RHFControlledComponents.Input
              id={`${field.id}-key`}
              componentId=""
              name={`tagsFilter.${index}.key`}
              control={control}
              aria-label={formatMessage({
                defaultMessage: 'Key',
                description: '',
              })}
              placeholder={formatMessage({
                defaultMessage: 'Key',
                description: '',
              })}
              required
            />
            <RHFControlledComponents.LegacySelect
              id={`${field.id}-op`}
              name={`tagsFilter.${index}.operator`}
              control={control}
              options={OPERATORS.map((op) => ({ key: op, value: op }))}
              aria-label={formatMessage({
                defaultMessage: 'Operator',
                description: '',
              })}
              css={{ minWidth: '14ch' }}
            />
            <RHFControlledComponents.Input
              id={`${field.id}-value`}
              componentId=""
              name={`tagsFilter.${index}.value`}
              control={control}
              aria-label={formatMessage({
                defaultMessage: 'Value',
                description: '',
              })}
              placeholder={formatMessage({
                defaultMessage: 'Value',
                description: '',
              })}
              required
            />
            <Button
              componentId=""
              type="tertiary"
              onClick={() => remove(index)}
              disabled={fields.length === 1}
              aria-label={formatMessage({
                defaultMessage: 'Remove filter',
                description: 'Button to remove a filter in the tags filter popover for experiments page search by tags',
              })}
            >
              <CloseIcon />
            </Button>
          </Fragment>
        ))}
      </fieldset>
      <div css={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button componentId="" onClick={() => append(EMPTY_TAG)} icon={<PlusIcon />}>
          Add filter
        </Button>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <Button componentId="" onClick={() => onSave([])}>
            Clear filters
          </Button>
          <Button htmlType="submit" componentId="" type="primary">
            Apply filters
          </Button>
        </div>
      </div>
    </form>
  );
}
