# Radiologist dashboard — static SPA + nginx reverse proxy (dashboard API, BioViL, Orthanc)
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENV ORTHANC_USER=manthana
ENV ORTHANC_PASS=manthana-pacs-secret
EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.sh"]
