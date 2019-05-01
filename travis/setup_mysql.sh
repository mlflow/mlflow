#!/usr/bin/env bash
set -ex

mysql --version
export MYSQL_TEST_USERNAME="root"
export MYSQL_TEST_PASSWORD="password"
sudo mysql -e "use mysql; update user set authentication_string=PASSWORD('password') where User='root'; update user set plugin='mysql_native_password';FLUSH PRIVILEGES;"
sudo mysql_upgrade -u "$MYSQL_TEST_USERNAME" -p"$MYSQL_TEST_PASSWORD"
sudo service mysql restart

set +ex
