import React from 'react';
import {
  useLocation,
  useNavigate,
  useParams,
  useNavigationType,
  Location,
  NavigateFunction,
  Params as RouterDOMParams,
  NavigationType,
} from 'react-router-dom-v5-compat';

export interface WithRouterNextProps<Params extends RouterDOMParams> {
  navigate: NavigateFunction;
  location: Location;
  params: Params;
  navigationType: NavigationType;
}

/**
 * This HoC serves as a retrofit for class components enabling them to use
 * react-router v6's location, navigate and params being injected via props.
 */
export const withRouterNext =
  <Props, Params extends RouterDOMParams>(
    Component: React.ComponentType<Props & WithRouterNextProps<Params>>,
  ) =>
  (props: Omit<Props, 'location' | 'navigate' | 'params' | 'navigationType'>) => {
    const location = useLocation();
    const navigate = useNavigate();
    const navigationType = useNavigationType();
    const params = useParams<Params>();

    return (
      <Component
        params={params as Params}
        location={location}
        navigate={navigate}
        navigationType={navigationType}
        {...(props as Props)}
      />
    );
  };
