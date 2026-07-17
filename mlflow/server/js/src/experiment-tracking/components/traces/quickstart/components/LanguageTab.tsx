import { SegmentedControlButton, SegmentedControlGroup } from '@databricks/design-system';

export type Language = 'python' | 'typescript' | 'opentelemetry';

export const LanguageTab = ({
  language,
  setLanguage,
}: {
  language: Language;
  setLanguage: (lang: Language) => void;
}) => {
  const tabs: { key: Language; label: string }[] = [
    { key: 'python', label: 'Python' },
    { key: 'typescript', label: 'TypeScript' },
    { key: 'opentelemetry', label: 'OpenTelemetry' },
  ];

  return (
    <SegmentedControlGroup
      name="mlflow.traces.onboarding.language-selector"
      componentId="mlflow.traces.onboarding.language_selector"
      value={language}
      onChange={(event) => setLanguage(event.target.value as Language)}
    >
      {tabs.map(({ key, label }) => (
        <SegmentedControlButton key={key} value={key}>
          {label}
        </SegmentedControlButton>
      ))}
    </SegmentedControlGroup>
  );
};
