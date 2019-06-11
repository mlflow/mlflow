## What changes are proposed in this pull request?
 
The possibility of binding gunicorn to a socket instead of a TCP host:port tuple
while serving a model. This allows to serve a model through HTTPS with Nginx 
 
## How is this patch tested?
 
A new cli option is added to mlflow models serve to server the model through a gunicorn socket:
```bash
mlflow models serve -m path_to_model -s socket_name.sock
```

This is thought to be used with Nginx and HTTPS protocol, but it can be easily
tested with the following command (for the Tutorial example):
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"coluhlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' --unix-socket /path_to_your_socket/socket_name.sock http://localhost/invocations
```

## Release Notes
 
### Is this a user-facing change? 

- [ ] No. You can skip the rest of this section.
- [X] Yes. Give a description of this change to be included in the release notes for MLflow users.
 
Explanation in the previous description.
 
### What component(s) does this PR affect?
 
- [ ] UI
- [X] CLI 
- [ ] API 
- [ ] REST-API 
- [ ] Examples 
- [ ] Docs
- [ ] Tracking
- [ ] Projects 
- [ ] Artifacts 
- [X] Models 
- [ ] Scoring 
- [ ] Serving
- [ ] R
- [ ] Java
- [ ] Python

### How should the PR be classified in the release notes? Choose one:
 
- [ ] `rn/breaking-change` - The PR will be mentioned in the "Breaking Changes" section
- [ ] `rn/none` - No description will be included. The PR will be mentioned only by the PR number in the "Small Bugfixes and Documentation Updates" section
- [X] `rn/feature` - A new user-facing feature worth mentioning in the release notes
- [ ] `rn/bug-fix` - A user-facing bug fix worth mentioning in the release notes
- [ ] `rn/documentation` - A user-facing documentation change worth mentioning in the release notes
