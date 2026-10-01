import { InfoSmallIcon, Popover } from '@databricks/design-system';
import { FormattedMessage, defineMessage, useIntl } from 'react-intl';

const tooltipIntroMessage = defineMessage({
  defaultMessage:
    'Free-text search matches name and description through {searchText}. Structured filters are sent as {filterString} clauses.',
  description: 'Tooltip explaining Skill Registry catalog search',
});

export const SkillSearchInputHelpTooltip = () => {
  const { formatMessage } = useIntl();
  const labelText = formatMessage(tooltipIntroMessage, { searchText: 'search_text', filterString: 'filter_string' });

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
          <FormattedMessage
            {...tooltipIntroMessage}
            values={{ searchText: <b>search_text</b>, filterString: <b>filter_string</b> }}
          />
          <br />
          <br />
          <FormattedMessage
            defaultMessage="Examples:"
            description="Text header for examples of Skill Registry search syntax"
          />
          <br />• search_text ILIKE &quot;%review%&quot;
          <br />• status = &quot;active&quot; AND organization = &quot;acme&quot;
          <br />• source_type = &quot;git&quot;
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};
