import { describe, test, expect } from '@jest/globals';
import { createIntl, createIntlCache } from 'react-intl';
import enMessages from '../../../lang/default/en.json';
import type { BudgetPolicy } from '../../types';
import { formatBudgetPolicySummary, formatDuration, formatOnExceeded } from './budgetFormatUtils';

const intl = createIntl({ locale: 'en' }, createIntlCache());

// Message ids are content hashes, so a translated catalog has to look them up in the extracted messages.
const messageId = (defaultMessage: string, description: string) => {
  const entry = Object.entries(enMessages as Record<string, { defaultMessage: unknown; description?: string }>).find(
    ([, message]) => message.defaultMessage === defaultMessage && message.description === description,
  );
  if (!entry) {
    throw new Error(`No extracted message "${defaultMessage}" (${description})`);
  }
  return entry[0];
};

describe('formatDuration', () => {
  test.each([
    [1, 'MINUTES' as const, '1 Minute'],
    [15, 'MINUTES' as const, '15 Minutes'],
    [1, 'HOURS' as const, 'Hourly'],
    [6, 'HOURS' as const, '6 Hours'],
    [1, 'DAYS' as const, 'Daily'],
    [2, 'WEEKS' as const, '2 Weeks'],
    [1, 'MONTHS' as const, 'Monthly'],
    [1000, 'MONTHS' as const, '1,000 Months'],
  ])('formats %d %s as %s', (value, unit, expected) => {
    expect(formatDuration(intl, value, unit)).toBe(expected);
  });
});

describe('formatOnExceeded', () => {
  test.each([
    ['ALERT' as const, 'Alert'],
    ['REJECT' as const, 'Reject'],
  ])('formats %s as %s', (action, expected) => {
    expect(formatOnExceeded(intl, action)).toBe(expected);
  });
});

describe('formatBudgetPolicySummary', () => {
  test('combines the amount, the reset period, and the action', () => {
    const policy: BudgetPolicy = {
      budget_policy_id: 'bp-1',
      budget_unit: 'USD',
      budget_amount: 100,
      duration: { unit: 'MINUTES', value: 15 },
      target_scope: 'GLOBAL',
      budget_action: 'REJECT',
      created_at: 0,
      last_updated_at: 0,
    };

    expect(formatBudgetPolicySummary(intl, policy)).toBe('$100 / 15 Minutes — Reject');
  });
});

describe('with a translated catalog', () => {
  const translatedIntl = createIntl(
    {
      locale: 'fr',
      defaultLocale: 'en',
      messages: {
        [messageId('Hourly', 'Budget reset period of 1 hour')]: 'Toutes les heures',
        [messageId('{count, plural, one {Minute} other {Minutes}}', 'Budget reset period unit')]:
          '{count, plural, one {minute} other {minutes}}',
        [messageId('Reject', 'Budget policy action that rejects requests')]: 'Rejeter',
      },
      // The catalog above is partial on purpose; untranslated messages fall back to English.
      onError: () => {},
    },
    createIntlCache(),
  );

  test('shows the translated labels', () => {
    expect(formatDuration(translatedIntl, 1, 'HOURS')).toBe('Toutes les heures');
    expect(formatDuration(translatedIntl, 15, 'MINUTES')).toBe('15 minutes');
    expect(formatOnExceeded(translatedIntl, 'REJECT')).toBe('Rejeter');
  });
});
