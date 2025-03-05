import React from 'react';
import {
  type Location,
  type Params as RouterDOMParams,
  useLocation,
  useNavigate,
  useParams,
} from './RoutingUtils';

export interface WithRouterNextProps<Params extends RouterDOMParams = RouterDOMParams> {
  navigate: ReturnType<typeof useNavigate>;
  location: Location;
  params: Params;
}

/**
 * This HoC serves as a retrofit for class components enabling them to use
 * react-router v6's location, navigate, and params being injected via props.
 */
export const withRouterNext =
  <T, Props extends WithRouterNextProps>(
    Component: React.ComponentType<T & WithRouterNextProps>,
  ) =>
  (props: Omit<Props, keyof WithRouterNextProps>) => {
    const location = useLocation();
    const navigate = useNavigate();
    const params = useParams<RouterDOMParams>();

    return (
      <Component
        {...(props as Props)}
        params={params as RouterDOMParams}
        location={location}
        navigate={navigate}
      />
    );
  };