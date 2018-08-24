#!/bin/bash

mlflow server --host 0.0.0.0  2>&1 | tee server.log
