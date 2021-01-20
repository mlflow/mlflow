#!/bin/bash

# Because $OSTYPE could be cygwin, msys, or win32, etc on Windows. The possibilities are endless.
if [[ "${OSTYPE}" != linux* && "${OSTYPE}" != darwin* ]]; then
  R_HOME='/c/R'
  export PATH="${PATH}:${R_HOME}/bin"
fi
