import React from 'react';

import {
  type Location,
  type Params as RouterDOMParams,
  type NavigateOptions,
  type To,
  useLocation,
  useNavigate,
  useParams,
} from './RoutingUtils';
import { useSearchParams } from './RoutingUtils';

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
    Props extends JSX.IntrinsicAttributes &
      JSX.LibraryManagedAttributes<React.ComponentType<React.PropsWithChildren<T>>, React.PropsWithChildren<T>>,
    Params extends RouterDOMParams = RouterDOMParams,
  >(
    Component: React.ComponentType<React.PropsWithChildren<T>>,
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
    const [searchParams, setSearchParams] = useSearchParams();

    return (
      <Component
        /* prettier-ignore */
        params={params as Params}
        location={location}
        navigate={navigate}
        searchParams={searchParams}
        setSearchParams={setSearchParams}
        {...(props as Props)}
      />
    );
  };
