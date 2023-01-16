# Establish SSH to localhost for test_sftp_artifact_repo.py
ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa

cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
ssh-keyscan -H localhost >> ~/.ssh/known_hosts
ssh $(whoami)@localhost exit
export LOGNAME=$(whoami)
