#Install mlflow service
#Run in sudo
docker build . -t mlflow -f Dockerfile.srv

#Using folowing commands to install service
  #mv mlflowd.service /etc/systemd/system/
  #systemctl reload-daemon
  #systemctl enable mlflowd
  #systemctl start mlflowd
