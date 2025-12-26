/**
 * This file is the only one that should directly import from 'react-router-dom' module
 */
/* eslint-disable no-restricted-imports */

import type { ComponentProps } from 'react';
import {
  generatePath,
  useParams as useParamsDirect,
  useLocation as useLocationDirect,
  Link as LinkDirect,
} from 'react-router-dom';

import { Typography } from '@databricks/design-system';

const useParams = useParamsDirect;
const useLocation = useLocationDirect;

const Link = LinkDirect;

export const createMLflowRoutePath = (routePath: string) => {
  return routePath;
};

export { generatePath, useParams, useLocation, Link };
