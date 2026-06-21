FROM node:20-alpine

WORKDIR /app

COPY app/frontend/package*.json ./
RUN npm install

COPY app/frontend ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
