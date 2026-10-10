-- Runs once on first start of the Postgres container.
CREATE DATABASE mlflow;
\connect coldchain
CREATE EXTENSION IF NOT EXISTS timescaledb;
