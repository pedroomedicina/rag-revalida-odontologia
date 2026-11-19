"""Postgres full-text search setup shared by the index builder and retrieval.

Two problems this module fixes in keyword retrieval:

- `plainto_tsquery` ANDs every word of the question, so a natural question only matched
  chunks containing *all* of its content words (including "endodoncia", "cuándo", ...).
  Keyword search now ORs the question's lexemes; `ts_rank_cd` still ranks chunks that
  match more of them higher.
- The stock `spanish` config is accent-sensitive in practice ("medicacion" and
  "medicación" stem differently). `spanish_unaccent` strips accents before stemming,
  on both the indexed text and the question.
"""

from __future__ import annotations

TS_CONFIG = "spanish_unaccent"

SEARCH_TSV_EXPRESSION = (
    f"to_tsvector('{TS_CONFIG}'::regconfig, "
    "coalesce(title, '') || ' ' || coalesce(section_title, '') || ' ' || coalesce(text, ''))"
)

# OR of the question's lexemes. to_tsvector applies the same parsing, stopwords, unaccent,
# and stemming as the index; quote_literal + ::tsquery keeps each lexeme as-is (no second
# stemming pass). A question made only of stopwords yields NULL, which matches nothing.
KEYWORD_TSQUERY_SQL = (
    "(SELECT string_agg(quote_literal(lexeme), ' | ') "
    f"FROM unnest(tsvector_to_array(to_tsvector('{TS_CONFIG}'::regconfig, %s))) AS lexeme"
    ")::tsquery"
)


def ensure_text_search_config(cursor) -> None:
    """Create the unaccent extension and the spanish_unaccent config if missing."""
    cursor.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
    cursor.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = '{TS_CONFIG}') THEN
                CREATE TEXT SEARCH CONFIGURATION {TS_CONFIG} (COPY = spanish);
                ALTER TEXT SEARCH CONFIGURATION {TS_CONFIG}
                    ALTER MAPPING FOR hword, hword_part, word WITH unaccent, spanish_stem;
            END IF;
        END
        $$;
        """
    )


def search_tsv_expression_of(cursor, table: str) -> str | None:
    """Return the generated-column expression of table.search_tsv, or None if absent."""
    cursor.execute(
        """
        SELECT pg_get_expr(d.adbin, d.adrelid)
        FROM pg_attrdef d
        JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
        WHERE d.adrelid = to_regclass(%s) AND a.attname = 'search_tsv'
        """,
        (table,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def ensure_search_column(cursor, table: str) -> bool:
    """Make table.search_tsv use SEARCH_TSV_EXPRESSION, rebuilding it if needed.

    Existing indexes were created with the plain `spanish` config; a generated column
    cannot be altered in place, so it is dropped (with its GIN index) and re-added.
    Returns True when the column was (re)created.
    """
    current = search_tsv_expression_of(cursor, table)
    if current is not None and TS_CONFIG in current:
        return False
    if current is not None:
        cursor.execute(f"ALTER TABLE {table} DROP COLUMN search_tsv;")
    cursor.execute(
        f"ALTER TABLE {table} ADD COLUMN search_tsv tsvector "
        f"GENERATED ALWAYS AS ({SEARCH_TSV_EXPRESSION}) STORED;"
    )
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_search_tsv_gin_idx ON {table} USING gin (search_tsv);"
    )
    return True
