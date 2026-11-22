import { InfoSmallIcon, Popover } from '@databricks/design-system';
import { FormattedMessage, defineMessage, useIntl } from 'react-intl';

const tooltipIntroMessage = defineMessage({
  defaultMessage:
    'Type text to search skill names and descriptions. To search by tags or by names and tags,{newline}use a simplified version of the SQL {whereBold} clause.',
  description: 'Tooltip explaining how to search skills in the registry',
});

export const SkillSearchInputHelpTooltip = () => {
  const { formatMessage } = useIntl();
  const labelText = formatMessage(tooltipIntroMessage, { newline: ' ', whereBold: 'WHERE' });

  return (
    <Popover.Root componentId="mlflow.skill_registry.search.help_tooltip">
      <Popover.Trigger
        aria-label={labelText}
        css={{ border: 0, background: 'none', padding: 0, lineHeight: 0, cursor: 'pointer' }}
      >
        <InfoSmallIcon />
      </Popover.Trigger>
      <Popover.Content align="start">
        <div>
          <FormattedMessage {...tooltipIntroMessage} values={{ newline: <br />, whereBold: <b>WHERE</b> }} />
          <br />
          <br />
          <FormattedMessage
            defaultMessage="Examples:"
            description="Text header for examples of Skill Registry search syntax"
          />
          <br />• tags.team = &quot;platform&quot;
          <br />• name ILIKE &quot;%review%&quot; AND tags.team = &quot;platform&quot;
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};
