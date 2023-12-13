# unsuperved-kg



## Macrostrat DB explorer

The `macrostrat_db/database_explorer.ipynb` contains code to explore the database dump of the macrostrat datbase. It produces two files, a `all_columns.csv` which contains metadata
about all the columns and tables in the `macrostrat` schema and `macrostrat_graph.csv` which contains data for a graph about units metadata extracted from the query:

```
SELECT *
FROM units u
JOIN unit_liths ul
  ON u.id = ul.unit_id
JOIN liths l
  ON l.id = ul.lith_id
 -- Linking table between unit lithologies and their attributes
 -- (essentially, adjectives describing that component of the rock unit)
 -- Examples: silty mudstone, dolomitic sandstone, mottled shale, muscovite garnet granite
JOIN unit_lith_atts ula
  ON ula.unit_lith_id = ul.id
JOIN lith_atts la
  ON ula.lith_att_id = la.id
```

## REBEL based knowledgre graph extraction

To extract relationships from the text corpus, we utilize the REBEL model: [https://github.com/Babelscape/rebel](https://github.com/Babelscape/rebel) which is a seq2sel model for relationship extraction.
