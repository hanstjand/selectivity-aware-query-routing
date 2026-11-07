-- Indexes, created after the bulk load. Each statement is timed separately by the loader.
CREATE INDEX {table}_similarity_btree ON {table} (similarity);
CREATE INDEX {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction});
