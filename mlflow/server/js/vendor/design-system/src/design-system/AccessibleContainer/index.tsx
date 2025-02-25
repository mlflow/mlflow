import { visuallyHidden } from '../utils';

export interface AccessibleContainerProps {
  children: React.ReactNode;
  label?: React.ReactNode;
}

export function AccessibleContainer({ children, label }: AccessibleContainerProps) {
  if (!label) {
    return <>{children}</>;
  }

  return (
    <div css={{ cursor: 'progress' }}>
      <span css={visuallyHidden}>{label}</span>
      <div aria-hidden>{children}</div>
    </div>
  );
}
