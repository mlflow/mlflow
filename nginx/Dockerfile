FROM nginx:1.17.6
# Remove default Nginx config
RUN rm /etc/nginx/nginx.conf
# Copy the modified Nginx conf
COPY nginx.conf /etc/nginx
# Copy proxy config
COPY mlflow.conf /etc/nginx/sites-enabled/