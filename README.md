# MLflow: A Machine Learning Lifecycle Platform

This is the Nagra fork readme. README.rst is the official one.

The idea is to contribute to MLFlow as much as possible, but while PRs are reviewed,
this repo is used to build the custome version of MLFlow used by Insight.

## Branch policy

The "master" branch is `origin/ni-master`. `origin/ni-master` is the version that is released and deployed in Insight.

The branch `origin/master` should always be in sync with `upstream/master` (where `upstream` is the official MLFlow repo).

New pull requests to the official MLFlow repo must be created from feature branches created from `origin/repo`.

Features that are not yet merged into the official MLFlow repo can be merged into `ni-master` in order to be released internally.

## Release

Release must always be done from branch `ni-master`.

```
git checkout ni-master
python setup.py bdist_wheel
python3 -m twine upload \
    --user insight-rw \
    --password $(gopass nexus/insight-rw) \
    --repository-url https://nexus.infra.nagra-insight.com/repository/pypi-dev/ \
    dist/*
```
