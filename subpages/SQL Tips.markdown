---
layout: page
title: SQL Quick Reference Guide
permalink: /sql_tips
---

## Purpose
I'm putting together a list of helpful code snippets and links for SQL.

### SQL is:
``` sql
-- Written: 
SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY

-- Processes: 
FROM, WHERE, GROUP BY, HAVING, SELECT, ORDER BY
```

### Window Functions
``` sql
-- Row Numbers for First/Last or Highest/Lowest Values:
ROW_NUMBER() OVER (PARTITION BY _ ORDER BY _ ) as rn

-- Next and Last Values: 
LEAD(c.return_value, offset_val) OVER(PARTITION BY _ ORDER BY _ ) as next_val
LAG(c.return_value, offset_val) OVER(PARTITION BY _ ORDER BY _ ) as last_val

-- Percent Rank:
PCT_RANK()OVER(PARTITION BY _ ORDER BY _) as pct_rank
```

### String Manipulation
[Postgres Documentation](https://www.postgresql.org/docs/current/functions-matching.html)

[Regex Pattern Testing](https://regexr.com/)

``` sql
-- Extract Exact Part of String:
SUBSTRING(c.col, start_pos, num_extract)
SPLIT_PART(c.col, 'split_string', part_num)

-- Find One Column's String in Another Column's String:
where CHARINDEX((select a.name from a_table a), c.list_o_names) > 0

-- LIKE Function May (or may not) Take Into Account Capitalization:
c.col LIKE '%string%' = c.col ~ 'string'
c.col LIKE '%string%' != c.col ~* 'string'
```

### String Aggregation
Use LISTAGG function to take multiple string rows (c.string) with the same key column (c.key_col) into one aggregated string column (cte.agg_string) in a CTE, then re-join with other tables on the key column.

``` sql
with cte as (
select
  c.key_col
, LISTAGG(c.string, ', ') WITHIN GROUP (ORDER BY c.string) agg_string
from string_table c
group by c.key_col
)
```


### Location
[PostGIS Function Documentation](https://postgis.net/docs/PostGIS_Special_Functions_Index.html)

You can use [kepler](https://kepler.gl) to visualize geoJSON. Popular geoJSON/geometry/geography objects include points, lines, and polygons.

``` sql
-- Output geoJSON:
ST_AsGeoJSON(c.polygon_col) -- Output a geoJSON 

-- Make Geometry/Geography Objects:
ST_GeomFromText(long lat, long lat, ...) -- Make Geom from a List of Coordinates
ST_MakePoint(long,lat) -- Make Geom Point
ST_LineFromText('LINESTRING(long lat, long lag, ...)') -- Make Line from a List of Coordinates
ST_MakePOLYgon(ST_GeomFromText('LINESTRING(long lat, long lat, ...)')) -- Make Polygon from a List of Coordinates

-- Find:
ST_Length(c.line) -- Length
ST_Area(c.polygon) -- Area
ST_Centroid(c.polygon) -- Centroid
ST_StartPoint(c.line) -- Start Point
ST_EndPoint(c.line) -- End Point

-- Find if Two Geographies Intersect: 
ST_Intersects(c.geom, c.geom) -- T/F if a Geom Intersects Another Geom
ST_Within(c.geom, c.geom) -- T/F if a one Geom is in Another Geom
ST_Overlaps(c.geom, c.geom) -- T/F if a Geom Overlaps but is Not Contained in Another Geom
ST_Distance(c.geom, c.geom, 'FALSE') -- Spherical Distance Between Two Geoms (Meters)
ST_Distance(c.geom, c.geom, 'TRUE') -- Shperoid Distance Between Two Geoms (Meters)
```

### JSON
Extract JSON Objects from a JSON Column

``` sql
Json_extract_path_text(c.json, 'key') -- Redshift
c.json::json ->> 'key' -- Postgres
```

### Datetime
Assume UTC

[Postgres Documentation](https://postgresql.org/docs/current/functions-datetime.html)

[Redshift Documentation](https://docs.aws.amazon.com/redshift/latest/dg/Date_functions_header.html)
and [Redshift Datetime Parts](https://docs.aws.amazon.com/redshift/latest/dg/r_FORMAT_strings.html)

``` sql
-- Current Timestamp:
NOW() or GETDATE()

-- Current Date (no Timestamp or 00:00:00 ts):
CURRENT_DATE

-- Make Timestamp Columns:
c.col::ts -- Force Timestamp (might not work if column is not a general timestamp format)
TO_TIMESTAMP (c.col, 'format') -- Specify Timestamp Parts (see datetime parts for format formatting)

-- Make Date Columns:
c.timestamp::date
DATE(c.timestamp) -- Date of Timestamp
	-- e.g .c.timestamp = '2022-08-01 10:00:00'
	DATE(c.timestamp) = '2022-08-01'::date

-- Specify Interval:
INTERVAL 'string interval' 
	-- e.g. 
	c.col::ts - INTERVAL '2 days' -- Will Give the Timestamp Column Minus 2 Days
	
-- Truncate and Get Certain Timestamp Parts:
DATE_TRUNC('day', c.timestamp) -- Returns the Timestamp up to the Day
DATE_PART('day', c.timestamp) -- Returns the Day Date Part of the Timestamp
	-- e.g. c.timestamp = '2022-08-01 10:00:00'
	DATE_TRUNC('day', c.timestamp) = '2022-08-01 00:00:00'::ts
	DATE_PART('day', c.timestamp) = 1

-- Extract Numeric Time Difference (in hours):
EXTRACT(epoch FROM (end_time - start_time)) / 3600
```

### Various Functions
#### Coalesce
Return the first non-null value. An example of adding timezones to UTC timezone columns is below.

``` sql
c.col::ts + COALESCE(tz.tz_offset, '-07:00:00') -- Returns Timezone (if specified in the tz table, or UTC -7 if null)
```

#### Null If
Return NULL if a column value matches another value. This example assumes some c.val_2 rows are 0, which makes SQL throw division errors.

``` sql
select
c.val_1::float / NULLIF((c.val_2/20)::float, 0)
from div_table c
```

#### Pivot
Pivot a table by a row and column to column and value respectively. Works in Athena SQL.

``` sql
select
metric['a'] metric_a
, metric['b'] metric_b
, metric['c'] metric_c
from (
select
map_agg(metric, value) metric
from unpivot
)
```