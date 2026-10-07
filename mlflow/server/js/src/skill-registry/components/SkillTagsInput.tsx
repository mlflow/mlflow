import { useState } from 'react';
import { Button, Input, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { KeyValueTag } from '../../common/components/KeyValueTag';

export const SkillTagsInput = ({
  tags,
  onChange,
}: {
  tags: Record<string, string>;
  onChange: (tags: Record<string, string>) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const keyLabel = intl.formatMessage({ defaultMessage: 'Key', description: 'Aria label for a skill tag key' });
  const valueLabel = intl.formatMessage({ defaultMessage: 'Value', description: 'Aria label for a skill tag value' });

  const addTag = () => {
    const trimmedKey = key.trim();
    if (!trimmedKey) return;
    onChange({ ...tags, [trimmedKey]: value });
    setKey('');
    setValue('');
  };

  const removeTag = (tagKey: string) => {
    const { [tagKey]: _removed, ...rest } = tags;
    onChange(rest);
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Tags" description="Label for skill tags saved after registration" />
      </Typography.Text>
      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
        <Input
          componentId="mlflow.skill_registry.register_modal.tag_key"
          aria-label={keyLabel}
          placeholder={keyLabel}
          value={key}
          onChange={(event) => setKey(event.target.value)}
          css={{ flex: 1 }}
        />
        <Input
          componentId="mlflow.skill_registry.register_modal.tag_value"
          aria-label={valueLabel}
          placeholder={valueLabel}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          css={{ flex: 1 }}
        />
        <Button
          componentId="mlflow.skill_registry.register_modal.tag_add"
          icon={<PlusIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Add tag',
            description: 'Aria label for adding a skill tag',
          })}
          disabled={!key.trim()}
          onClick={addTag}
        />
      </div>
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="Key/value metadata stored with the skill. You can edit it later from the skill page."
          description="Hint for skill registration tags"
        />
      </Typography.Hint>
      {Object.keys(tags).length > 0 && (
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
          {Object.entries(tags).map(([tagKey, tagValue]) => (
            <KeyValueTag
              key={tagKey}
              isClosable
              tag={{ key: tagKey, value: tagValue }}
              onClose={() => removeTag(tagKey)}
            />
          ))}
        </div>
      )}
    </div>
  );
};
