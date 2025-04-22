# How to sync the `mlflow-3` branch with the `master` branch manually

## Steps

1. Run the following commands to prepare a sync branch:

   ```bash
   # Sync the local `master` branch with the remote `master` branch
   git checkout master
   git pull upstream master

   # Sync the local `mlflow-3` branch with the remote `mlflow-3` branch
   git checkout mlflow-3
   git pull upstream mlflow-3

   # Cut a new branch from the `mlflow-3` branch
   git checkout -b mlflow-3-sync mlflow-3

   # Merge the `master` branch into the sync branch
   git merge master
   # Resolve any conflicts
   git add .
   git merge --continue

   # Push the sync branch
   git push origin mlflow-3-sync
   ```

2. Once the sync branch is pushed, create a pull request from the sync branch to the 🚨 `mlflow-3` 🚨 branch.
3. Once the PR is reviewed and approved, merge it with the 🚨 `Create a merge commit` 🚨 option.
