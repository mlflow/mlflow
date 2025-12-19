import {
  useTagAssignmentForm,
  TagAssignmentRoot,
  TagAssignmentRow,
  TagAssignmentLabel,
  TagAssignmentKey,
  TagAssignmentValue,
  TagAssignmentRemoveButton,
} from '@databricks/web-shared/unified-tagging';
import type { UseFormReturn } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import type { KeyValueEntity } from '../types';

const keyProperty = 'key';
const valueProperty = 'value';

interface Props {
  name: string;
  form: UseFormReturn<any>;
  initialTags?: KeyValueEntity[];
}

/**
 * A component used for displaying the unified tagging form.
 */
export const UnifiedTaggingForm = ({ form, name, initialTags }: Props) => {
  const intl = useIntl();

  const tagsForm = useTagAssignmentForm({
    name,
    emptyValue: { key: '', value: '' },
    keyProperty,
    valueProperty,
    form,
    defaultValues: initialTags,
  });

  return (
    <TagAssignmentRoot {...tagsForm}>
      <TagAssignmentRow>
        <TagAssignmentLabel>
          <FormattedMessage defaultMessage="Key" description="Tag assignment modal > Key label" />
        </TagAssignmentLabel>
        <TagAssignmentLabel>
          <FormattedMessage defaultMessage="Value" description="Tag assignment modal > Value label" />
        </TagAssignmentLabel>
      </TagAssignmentRow>

      {tagsForm.fields.map((field, index) => {
        return (
          <TagAssignmentRow key={field.id}>
            <TagAssignmentKey
              index={index}
              rules={{
                validate: {
                  unique: (value) => {
                    const tags = tagsForm.getTagsValues();
                    if (tags?.findIndex((tag) => tag[keyProperty] === value) !== index) {
                      return intl.formatMessage({
                        defaultMessage: 'Key must be unique',
                        description: 'Error message for unique key in tag assignment modal',
                      });
                    }
                    return true;
                  },
                  required: (value) => {
                    const tags = tagsForm.getTagsValues();
                    if (tags?.at(index)?.[valueProperty] && !value) {
                      return intl.formatMessage({
                        defaultMessage: 'Key is required if value is present',
                        description: 'Error message for required key in tag assignment modal',
                      });
                    }
                    return true;
                  },
                },
              }}
            />
            <TagAssignmentValue index={index} />
            <TagAssignmentRemoveButton index={index} componentId="endpoint-tags-section.remove-button" />
          </TagAssignmentRow>
        );
      })}
    </TagAssignmentRoot>
  );
};
