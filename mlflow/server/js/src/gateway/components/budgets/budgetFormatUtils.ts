import type { BudgetAction, BudgetPolicy, BudgetUnit, DurationUnit } from '../../types';

export function formatBudgetAmount(amount: number, budgetUnit: BudgetUnit, digits: number = 2): string {
  if (budgetUnit === 'USD') {
    return `$${amount.toLocaleString(undefined, { maximumFractionDigits: digits })}`;
  }
  return `${amount}`;
}

export function formatDuration(value: number, unit: DurationUnit): string {
  if (value === 1) {
    const friendlyLabels: Partial<Record<DurationUnit, string>> = {
      DAYS: 'Daily',
      WEEKS: 'Weekly',
      MONTHS: 'Monthly',
    };
    if (friendlyLabels[unit]) return friendlyLabels[unit]!;
  }
  const typeLabels: Record<DurationUnit, string> = {
    MINUTES: value === 1 ? 'Minute' : 'Minutes',
    HOURS: value === 1 ? 'Hour' : 'Hours',
    DAYS: value === 1 ? 'Day' : 'Days',
    WEEKS: value === 1 ? 'Week' : 'Weeks',
    MONTHS: value === 1 ? 'Month' : 'Months',
  };
  return `${value} ${typeLabels[unit] ?? unit}`;
}

export function formatOnExceeded(action: BudgetAction): string {
  const labels: Record<BudgetAction, string> = {
    ALERT: 'Alert',
    REJECT: 'Reject',
  };
  return labels[action];
}

export function formatBudgetPolicySummary(policy: BudgetPolicy): string {
  return `${formatBudgetAmount(policy.budget_amount, policy.budget_unit)} / ${formatDuration(policy.duration_value, policy.duration_unit)} — ${formatOnExceeded(policy.budget_action)}`;
}
