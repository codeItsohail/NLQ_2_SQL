CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    question TEXT,
    sql_query TEXT,
    answer JSONB,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    user_id INT,
    question TEXT,
    sql_generated TEXT,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);
