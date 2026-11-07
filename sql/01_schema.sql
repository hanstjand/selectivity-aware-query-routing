-- Schema for the filtered vector search study. {table} and {dim} are filled in from the config.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

DROP TABLE IF EXISTS {table};
CREATE TABLE {table} (
    id         integer PRIMARY KEY,            -- Qdrant point id == row index in vectors.npy
    similarity double precision NOT NULL,      -- LAION CLIP image-text similarity (payload field)
    embedding  vector({dim}) NOT NULL          -- float16 source values, stored losslessly as float32
);

-- Store vectors inline: a {dim}-d float32 row (~2.1 KB) would otherwise be TOASTed, the planner ignores
-- TOAST in its cost estimates (pgvector README), and exact scans would pay a hidden detoast cost.
ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE PLAIN;

-- Static benchmark data: statistics must not change between ANALYZE and the end of the experiment.
ALTER TABLE {table} SET (autovacuum_enabled = false);
