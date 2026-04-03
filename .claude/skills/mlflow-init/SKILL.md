---
name: mlflow-init
description: Sync fork with upstream master, create a new branch, start dev server, and open in browser
disable-model-invocation: true
---

# MLflow Init

Run these steps in order:

1. **Sync fork with upstream master:**
   ```bash
   git fetch upstream master
   git checkout master
   git merge upstream/master
   git push origin master
   ```

2. **Create a new branch with a random name** (use a short adjective-noun combo, e.g. `swift-falcon`, `bright-cedar`):
   ```bash
   # Generate a random branch name
   BRANCH_NAME=$(python3 -c "
   import random
   adjectives = ['swift','bright','calm','bold','keen','warm','cool','fair','glad','kind','neat','pure','wise','epic','fine']
   nouns = ['falcon','cedar','river','spark','flame','ridge','bloom','frost','coral','maple','stone','cloud','dawn','grove','trail']
   print(f'{random.choice(adjectives)}-{random.choice(nouns)}')
   ")
   git checkout -b "$BRANCH_NAME"
   ```
   Tell the user the branch name.

3. **Start the MLflow dev server:**
   ```bash
   nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &
   ```

4. **Wait for the frontend to be ready**, then open in browser:
   ```bash
   # Poll until the React dev server is up (max ~60 seconds)
   for i in $(seq 1 30); do
     if curl -s -o /dev/null -w '%{http_code}' http://localhost:3000 | grep -q '200'; then
       break
     fi
     sleep 2
   done
   open http://localhost:3000
   ```

5. Report the branch name and that the dev server is running at http://localhost:3000.
