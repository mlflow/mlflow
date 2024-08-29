import React from 'react';
import type { EmotionJSX } from '@emotion/react/types/jsx-namespace';

import {
  type Location,
  type Params as RouterDOMParams,
  type NavigateOptions,
  type To,
  useLocation,
  useNavigate,
  useParams,
} from '../../common/utils/RoutingUtils';

export interface WithRouterNextProps<Params extends RouterDOMParams = RouterDOMParams> {
  navigate: ReturnType<typeof useNavigate>;
  location: Location;
  params: Params;
}

/**
 * This HoC serves as a retrofit for class components enabling them to use
 * react-router v6's location, navigate and params being injected via props.
 */
export const withRouterNext =
  <
    T,
    Props extends EmotionJSX.IntrinsicAttributes &
      EmotionJSX.LibraryManagedAttributes<React.ComponentType<T>, React.PropsWithChildren<T>>,
    Params extends RouterDOMParams = RouterDOMParams,
  >(
    Component: React.ComponentType<T>,
  ) =>
  (
    props: Omit<
      Props,
      | 'location'
      | 'navigate'
      | 'params'
      | 'navigationType'
      /* prettier-ignore*/
    >,
  ) => {
    const location = useLocation();
    const navigate = useNavigate();
    const params = useParams<Params>();

    return (
      <Component
        /* prettier-ignore */
        params={params as Params}
        location={location}
        navigate={navigate}
        {...(props as Props)}
      />
    );
  };
