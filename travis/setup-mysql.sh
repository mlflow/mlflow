#!/usr/bin/env bash
set -ex

mysql --version
export MYSQL_TEST_USERNAME="root"
export MYSQL_TEST_PASSWORD="new_password"
sudo mysql -e "use mysql; update user set authentication_string=PASSWORD('new_password') where User='root'; update user set plugin='mysql_native_password';FLUSH PRIVILEGES;"
sudo mysql_upgrade -u "$MYSQL_TEST_USERNAME" -pnew_password
sudo service mysql restart

set +ex
