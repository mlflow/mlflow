import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Skeleton } from 'antd';
import { injectIntl } from 'react-intl';

// In lieu of the lack of React.lazy / Suspense, we're declaring a scoped variable here.
// Obviously not ideal since it breaks encapsulation in all sorts of ways, but hopefully this is
// only short-term.
let Plot;

export async function retry(fn, retries = 2, interval = 500) {
  try {
    return await fn();
  } catch (error) {
    if (retries > 0) {
      await new Promise((resolve) => setTimeout(resolve, interval));
      return retry(fn, retries - 1, interval);
    } else {
      throw new Error(error);
    }
  }
}

export class LazyPlotImpl extends Component {
  static propTypes = {
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  constructor() {
    super();
    this.state = {
      isLoading: !Plot,
      errorMessage: null,
    };
  }

  componentDidMount() {
    if (this.state.isLoading) {
      this.triggerFetch();
    }
  }

  async triggerFetch() {
    let errorMessage;

    if (!Plot) {
      try {
        Plot = await this.fetchPlotly();
      } catch (e) {
        errorMessage = e.message;
      }
    }

    this.setState({
      isLoading: false,
      errorMessage,
    });
  }

  async fetchPlotly() {
    try {
      const lazyLoadedChunk = await retry(async () => import('react-plotly.js'));
      return lazyLoadedChunk.default;
    } catch (e) {
      throw new Error(
        this.props.intl.formatMessage({
          defaultMessage: 'Failed to load chart.',
          description: 'Error message when the plotly javascript library fails to load.',
        }),
      );
    }
  }

  render() {
    const { isLoading, errorMessage } = this.state;
    const { intl, ...passThroughProps } = this.props;

    if (errorMessage) {
      return <div>{errorMessage}</div>;
    }

    if (isLoading && !Plot) {
      return <Skeleton active />;
    }

    return <Plot {...passThroughProps} />;
  }
}

export const LazyPlot = injectIntl(LazyPlotImpl);
