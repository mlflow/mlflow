import spinner from '../static/mlflow-spinner.png';
import type { Interpolation, Theme } from '@emotion/react';
import { keyframes } from '@emotion/react';

type Props = {
  showImmediately?: boolean;
};

export function Spinner({ showImmediately }: Props) {
  return (
    <div css={(theme) => styles.spinner(theme, showImmediately)}>
      <img alt="Page loading..." src={spinner} />
    </div>
  );
}

const styles = {
  spinner: (theme: Theme, immediate?: boolean): Interpolation<Theme> => ({
    width: 100,
    marginTop: 100,
    marginLeft: 'auto',
    marginRight: 'auto',

    img: {
      position: 'absolute',
      opacity: 0,
      top: '50%',
      left: '50%',
      width: theme.general.heightBase * 2,
      height: theme.general.heightBase * 2,
      marginTop: -theme.general.heightBase,
      marginLeft: -theme.general.heightBase,
      animation: `${keyframes`
          0% {
            opacity: 1;
          }
          100% {
            opacity: 1;
            -webkit-transform: rotate(360deg);
                transform: rotate(360deg);
            }
          `} 3s linear infinite`,
      animationDelay: immediate ? '0s' : '0.5s',
    },
  }),
};
