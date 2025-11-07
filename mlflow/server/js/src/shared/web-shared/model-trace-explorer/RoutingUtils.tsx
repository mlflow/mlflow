/**
 * This file is the only one that should directly import from 'react-router-dom' module
 */
/* eslint-disable no-restricted-imports */

import type { ComponentProps } from 'react';
import { generatePath, useParams as useParamsDirect, Link as LinkDirect } from 'react-router-dom';

import { Typography } from '@databricks/design-system';

const useParams = useParamsDirect;

const Link = LinkDirect;

export const createMLflowRoutePath = (routePath: string) => {
  return routePath;
};

export { generatePath, useParams, Link };
