import { useDesignSystemTheme } from '../../Hooks';
import { Spinner } from '../../Spinner';

export const LoadingSpinner = (props: any) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Spinner
      css={{
        display: 'flex',
        alignSelf: 'center',
        justifyContent: 'center',
        alignItems: 'center',
        height: theme.general.heightSm,
        width: theme.general.heightSm,
        '> span': { fontSize: 20 },
      }}
      {...props}
    />
  );
};
