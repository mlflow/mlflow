import { useFieldArray, useForm } from 'react-hook-form';
import type { TagFilter } from '../hooks/useTagsFilter';
import { OPERATORS } from '../hooks/useTagsFilter';
import { Button, CloseIcon, PlusIcon, RHFControlledComponents, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { Interpolation, Theme } from '@emotion/react';
import { Fragment } from 'react';

const EMPTY_TAG = { key: '', value: '', operator: 'IS' } satisfies TagFilter;

type Props = {
  tagsFilter: TagFilter[];
  setTagsFilter: (_: TagFilter[]) => void;
};

export function ExperimentListViewTagsFilter({ tagsFilter, setTagsFilter }: Props) {
  const { control, handleSubmit } = useForm<{ tagsFilter: TagFilter[] }>({
    defaultValues: { tagsFilter: tagsFilter.length === 0 ? [EMPTY_TAG] : tagsFilter },
  });
  const { fields, append, remove } = useFieldArray({ control, name: 'tagsFilter' });
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const labelStyles: Interpolation<Theme> = {
    fontWeight: theme.typography.typographyBoldFontWeight,
  };

  const labels = {
    key: formatMessage({
      defaultMessage: 'Key',
      description: 'Tag filter input for key field in the tags filter popover for experiments page search by tags',
    }),
    operator: formatMessage({
      defaultMessage: 'Operator',
      description: 'Tag filter input for operator field in the tags filter popover for experiments page search by tags',
    }),
    value: formatMessage({
      defaultMessage: 'Value',
      description: 'Tag filter input for value field in the tags filter popover for experiments page search by tags',
    }),
  };

  return (
    <form
      onSubmit={handleSubmit((data) => setTagsFilter(data.tagsFilter))}
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
          {labels.key}
        </label>
        <label htmlFor={`${fields[0].id}-op`} css={labelStyles}>
          {labels.operator}
        </label>
        <label htmlFor={`${fields[0].id}-value`} css={labelStyles}>
          {labels.value}
        </label>
        <label />
        {fields.map((field, index) => (
          <Fragment key={field.id}>
            <RHFControlledComponents.Input
              id={`${field.id}-key`}
              componentId={`mlflow.experiment_list_view.tag_filter.key_input_${index}`}
              name={`tagsFilter.${index}.key`}
              control={control}
              aria-label={labels.key}
              placeholder={labels.key}
              required
            />
            <RHFControlledComponents.LegacySelect
              id={`${field.id}-op`}
              name={`tagsFilter.${index}.operator`}
              control={control}
              options={OPERATORS.map((op) => ({ key: op, value: op }))}
              aria-label={labels.operator}
              css={{ minWidth: '14ch' }}
            />
            <RHFControlledComponents.Input
              id={`${field.id}-value`}
              componentId={`mlflow.experiment_list_view.tag_filter.value_input_${index}`}
              name={`tagsFilter.${index}.value`}
              control={control}
              aria-label={labels.value}
              placeholder={labels.value}
              required
            />
            <Button
              componentId={`mlflow.experiment_list_view.tag_filter.remove_filter_button_${index}`}
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
        <Button
          componentId="mlflow.experiment_list_view.tag_filter.add_filter_button"
          onClick={() => append(EMPTY_TAG)}
          icon={<PlusIcon />}
        >
          <FormattedMessage
            defaultMessage="Add filter"
            description="Button to add a new filter in the tags filter popover for experiments page search by tags"
          />
        </Button>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.experiment_list_view.tag_filter.clear_filters_button"
            onClick={() => setTagsFilter([])}
          >
            <FormattedMessage
              defaultMessage="Clear filters"
              description="Button to clear filters in the tags filter popover for experiments page search by tags"
            />
          </Button>
          <Button
            htmlType="submit"
            componentId="mlflow.experiment_list_view.tag_filter.apply_filters_button"
            type="primary"
          >
            <FormattedMessage
              defaultMessage="Apply filters"
              description="Button to apply filters in the tags filter popover for experiments page search by tags"
            />
          </Button>
        </div>
      </div>
    </form>
  );
}
