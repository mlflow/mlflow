import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactNbView from './ShowArtifactNbView';
import { Cell, Source } from "@nteract/presentational-components";
import {
    Output, StreamText, DisplayData
} from "@nteract/outputs";
import {
  emptyNotebook, appendCellToNotebook, makeCodeCell, fromJS
} from "@nteract/commutable";


test('ShowArtifactNbView - notebook one code cell', () => {
  const sourceCode = "source code";
  const codeCell = makeCodeCell({source: sourceCode});
  const notebook = appendCellToNotebook(emptyNotebook, codeCell);
  const wrapper = shallow(<ShowArtifactNbView
    path="fakepath"
    runUuid="fakerunuuid"
  />);
  wrapper.setState({immutableNotebook: notebook, loading: false, error: undefined});
  expect(wrapper.find(Source).text()).toEqual(sourceCode);
});


test('ShowArtifactNbView - complex notebook', () => {
  const notebook = fromJS({
    "cells": [
      {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
          "# markdown cell"
        ]
      },
      {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
          {
            "name": "stdout",
            "output_type": "stream",
            "text": [
              "code cell\n"
            ]
          }
        ],
        "source": [
          "print(\"code cell\")"
        ]
      },
      {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
          {
            "ename": "NameError",
            "evalue": "name 'error_cell' is not defined",
            "output_type": "error",
            "traceback": [
              "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
              "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
              "\u001b[0;32m<ipython-input-2-bf8729c32b9f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0merror_cell\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
              "\u001b[0;31mNameError\u001b[0m: name 'error_cell' is not defined"
            ]
          }
        ],
        "source": [
          "error_cell"
        ]
      },
      {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [
          {
            "data": {
              "text/plain": [
                "[<matplotlib.lines.Line2D at 0x7fc7c292e4a8>]"
              ]
            },
            "execution_count": 3,
            "metadata": {},
            "output_type": "execute_result"
          },
          {
            "data": {
              "image/png": "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\n",
              "text/plain": [
                "<Figure size 432x288 with 1 Axes>"
              ]
            },
            "metadata": {
              "needs_background": "light"
            },
            "output_type": "display_data"
          }
        ],
        "source": [
          "import matplotlib.pyplot as plt\n",
          "import numpy as np\n",
          "x = np.linspace(0, 10)\n",
          "plt.plot(x, np.sin(x))"
        ]
      },
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": []
      }
    ],
    "metadata": {
      "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
      },
      "language_info": {
        "codemirror_mode": {
          "name": "ipython",
          "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.6.8"
      }
    },
    "nbformat": 4,
    "nbformat_minor": 4
  });

  const wrapper = shallow(<ShowArtifactNbView
    path="fakepath"
    runUuid="fakerunuuid"
  />);
  wrapper.setState({immutableNotebook: notebook, loading: false, error: undefined});
  expect(wrapper.find(Cell)).toHaveLength(5);
  expect(wrapper.find(Output)).toHaveLength(4);
  expect(wrapper.find(Output).at(2).find(StreamText)).toHaveLength(1);
  expect(wrapper.find(Output).at(3).find(DisplayData)).toHaveLength(1);
});

