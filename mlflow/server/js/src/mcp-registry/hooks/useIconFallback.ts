import { useEffect, useState } from 'react';
import { sanitizeHref } from '../utils';

export const useIconFallback = (primarySrc: string | undefined, fallbackSrc: string | undefined) => {
  const [primaryFailed, setPrimaryFailed] = useState(false);
  const [fallbackFailed, setFallbackFailed] = useState(false);

  useEffect(() => {
    setPrimaryFailed(false);
  }, [primarySrc]);

  useEffect(() => {
    setFallbackFailed(false);
  }, [fallbackSrc]);

  const activeSrc =
    primarySrc && !primaryFailed
      ? primarySrc
      : fallbackSrc && !fallbackFailed && fallbackSrc !== primarySrc
        ? fallbackSrc
        : undefined;

  const onError = () => {
    if (activeSrc === primarySrc) {
      setPrimaryFailed(true);
    } else {
      setFallbackFailed(true);
    }
  };

  return { activeSrc, onError };
};
