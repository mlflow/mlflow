import { defineMessages, type IntlShape, type MessageDescriptor } from 'react-intl';
import type { BudgetAction, BudgetPolicy, BudgetUnit, DurationUnit } from '../../types';

/** Unit names, shared by the reset period selector and the policy list so their translations can't drift apart. */
export const durationUnitMessages: Record<DurationUnit, MessageDescriptor> = defineMessages({
  MINUTES: { defaultMessage: '{count, plural, one {Minute} other {Minutes}}', description: 'Budget reset period unit' },
  HOURS: { defaultMessage: '{count, plural, one {Hour} other {Hours}}', description: 'Budget reset period unit' },
  DAYS: { defaultMessage: '{count, plural, one {Day} other {Days}}', description: 'Budget reset period unit' },
  WEEKS: { defaultMessage: '{count, plural, one {Week} other {Weeks}}', description: 'Budget reset period unit' },
  MONTHS: { defaultMessage: '{count, plural, one {Month} other {Months}}', description: 'Budget reset period unit' },
});

const singlePeriodMessages: Partial<Record<DurationUnit, MessageDescriptor>> = defineMessages({
  HOURS: { defaultMessage: 'Hourly', description: 'Budget reset period of 1 hour' },
  DAYS: { defaultMessage: 'Daily', description: 'Budget reset period of 1 day' },
  WEEKS: { defaultMessage: 'Weekly', description: 'Budget reset period of 1 week' },
  MONTHS: { defaultMessage: 'Monthly', description: 'Budget reset period of 1 month' },
});

const budgetActionMessages: Record<BudgetAction, MessageDescriptor> = defineMessages({
  ALERT: { defaultMessage: 'Alert', description: 'Budget policy action that sends a webhook alert' },
  REJECT: { defaultMessage: 'Reject', description: 'Budget policy action that rejects requests' },
});

export function formatBudgetAmount(amount: number, budgetUnit: BudgetUnit, digits: number = 2): string {
  if (budgetUnit === 'USD') {
    return `$${amount.toLocaleString(undefined, { maximumFractionDigits: digits })}`;
  }
  return `${amount}`;
}

export function formatDuration(intl: IntlShape, value: number, unit: DurationUnit): string {
  const singlePeriodMessage = value === 1 ? singlePeriodMessages[unit] : undefined;
  if (singlePeriodMessage) {
    return intl.formatMessage(singlePeriodMessage);
  }
  return intl.formatMessage(
    {
      defaultMessage: '{count, number} {unit}',
      description: 'Budget reset period as a length and a unit, for example "15 Minutes"',
    },
    { count: value, unit: intl.formatMessage(durationUnitMessages[unit], { count: value }) },
  );
}

export function formatOnExceeded(intl: IntlShape, action: BudgetAction): string {
  return intl.formatMessage(budgetActionMessages[action]);
}

export function formatBudgetPolicySummary(intl: IntlShape, policy: BudgetPolicy): string {
  return intl.formatMessage(
    {
      defaultMessage: '{amount} / {duration} — {action}',
      description: 'One-line budget policy summary, for example "$100 / Monthly — Reject"',
    },
    {
      amount: formatBudgetAmount(policy.budget_amount, policy.budget_unit),
      duration: formatDuration(intl, policy.duration.value, policy.duration.unit),
      action: formatOnExceeded(intl, policy.budget_action),
    },
  );
}
