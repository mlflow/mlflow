import {
  FormUI,
  InfoSmallIcon,
  Input,
  SimpleSelect,
  SimpleSelectOption,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { BudgetDuration, DurationUnit } from '../../types';

export interface BudgetDurationFormValue {
  // Kept as the raw input string so an in-progress edit (e.g. a cleared field) isn't coerced.
  durationValue: string;
  durationUnit: DurationUnit;
}

export const DEFAULT_BUDGET_DURATION: BudgetDurationFormValue = { durationValue: '1', durationUnit: 'MONTHS' };

export const toBudgetDurationFormValue = ({ unit, value }: BudgetDuration): BudgetDurationFormValue => ({
  durationValue: String(value),
  durationUnit: unit,
});

// The backend fails to compute a window end past year 9999 (from ~96,400 months), and that
// failure stops budget tracking for every policy. 1,000 of any unit stays far below it.
export const MAX_BUDGET_DURATION_VALUE = 1000;

/** Returns the duration to submit, or `null` when the length isn't a whole number from 1 to the max. */
export const parseBudgetDuration = ({
  durationValue,
  durationUnit,
}: BudgetDurationFormValue): BudgetDuration | null => {
  const trimmed = durationValue.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const value = Number(trimmed);
  return value >= 1 && value <= MAX_BUDGET_DURATION_VALUE ? { unit: durationUnit, value } : null;
};

const DURATION_UNITS: DurationUnit[] = ['MINUTES', 'HOURS', 'DAYS', 'WEEKS', 'MONTHS'];

interface BudgetDurationFieldProps {
  componentId: string;
  value: BudgetDurationFormValue;
  onChange: (change: Partial<BudgetDurationFormValue>) => void;
}

/** The "Reset period" section of the create and edit budget policy forms. */
export const BudgetDurationField = ({ componentId, value, onChange }: BudgetDurationFieldProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isValid = parseBudgetDuration(value) !== null;
  // A blank or invalid length falls back to the plural form, which reads naturally next to any number.
  const count = Number(value.durationValue) || 0;

  const unitLabels: Record<DurationUnit, string> = {
    MINUTES: intl.formatMessage(
      { defaultMessage: '{count, plural, one {Minute} other {Minutes}}', description: 'Budget reset period unit' },
      { count },
    ),
    HOURS: intl.formatMessage(
      { defaultMessage: '{count, plural, one {Hour} other {Hours}}', description: 'Budget reset period unit' },
      { count },
    ),
    DAYS: intl.formatMessage(
      { defaultMessage: '{count, plural, one {Day} other {Days}}', description: 'Budget reset period unit' },
      { count },
    ),
    WEEKS: intl.formatMessage(
      { defaultMessage: '{count, plural, one {Week} other {Weeks}}', description: 'Budget reset period unit' },
      { count },
    ),
    MONTHS: intl.formatMessage(
      { defaultMessage: '{count, plural, one {Month} other {Months}}', description: 'Budget reset period unit' },
      { count },
    ),
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Reset period" description="Budget reset period label" />
        </Typography.Text>
        <Tooltip
          componentId={`${componentId}.reset-period-tooltip`}
          content={
            <FormattedMessage
              defaultMessage="<link>Learn more about budget time windows.</link>"
              description="Tooltip with link to budget time windows documentation"
              values={{
                link: (chunks: React.ReactNode) => (
                  <a
                    href="https://mlflow.org/docs/latest/genai/governance/ai-gateway/budget-alerts-limits#time-windows"
                    target="_blank"
                    rel="noopener noreferrer"
                    css={{ color: 'inherit', textDecoration: 'underline' }}
                  >
                    {chunks}
                  </a>
                ),
              }}
            />
          }
        >
          <InfoSmallIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
        </Tooltip>
      </div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div css={{ width: theme.spacing.md * 6 }}>
          <Input
            componentId={`${componentId}.duration-value`}
            aria-label={intl.formatMessage({
              defaultMessage: 'Reset period length',
              description: 'Accessible label for the budget reset period length input',
            })}
            value={value.durationValue}
            onChange={(e) => onChange({ durationValue: e.target.value })}
            type="number"
            min={1}
            max={MAX_BUDGET_DURATION_VALUE}
            step={1}
            validationState={isValid ? undefined : 'error'}
          />
        </div>
        <SimpleSelect
          id={`${componentId}.duration`}
          componentId={`${componentId}.duration`}
          label={intl.formatMessage({
            defaultMessage: 'Reset period unit',
            description: 'Accessible label for the budget reset period unit selector',
          })}
          value={value.durationUnit}
          onChange={({ target }) => onChange({ durationUnit: target.value as DurationUnit })}
        >
          {DURATION_UNITS.map((unit) => (
            <SimpleSelectOption key={unit} value={unit}>
              {unitLabels[unit]}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>
      {!isValid && (
        <FormUI.Message
          type="error"
          message={intl.formatMessage(
            {
              defaultMessage: 'Enter a whole number from 1 to {max, number}.',
              description: 'Validation error for an invalid budget reset period length',
            },
            { max: MAX_BUDGET_DURATION_VALUE },
          )}
        />
      )}
    </div>
  );
};
