---
layout: page
title: PostgreSQL and Querying Pokémon
permalink: /pgsql/pkmn
---

## Install (Lessons Learned)
I installed PostgreSQL directly from the website [here](https://www.postgresql.org). I also immediately downloaded the corresponding documentation from [here](https://www.postgresql.org/docs/). After installing in my local 'C:\Program Files' folder, I ran into an odd issue where I could not log into the default user in psql, which also happened to not be the postgre default user. After opening pgAdmin4 and reading the logs found in data/logs/, the solution was to add both a Database and Login/Group Role with the same name as the user defaulting in. This allowed me to finally 'log in' to psql. 

Overall, I found the overall query writing experience to be better using pgAdmin directly, than trying to write in psql command line. I have used MS SQL in the past, so the Query Tool within pgAdmin was very familiar. It is also useful see the tables and views directly in your database.

## Using Window Function to Find Pokémon First Type Ranking Based on Total Base Points
In order to accommodate multiple typing, I decided to do do two average window functions over the two types and then average the two results. Although a perfect measurement, averaging the total_points of the two types and then taking the average of that number still gives insight into how powerful a Pokémon is comparatively to other Pokémon with the same type 1 and type 2 typing. The results were then sorted by the first Pokémon type.

```sql
WITH pkmn_win as (SELECT pokedex_number, name, type_1, type_2, total_points,
AVG(total_points) OVER(PARTITION BY type_1) as avg_type1,
AVG(total_points) OVER(PARTITION BY type_2) as avg_type2,
rank() OVER (PARTITION BY type_1 ORDER BY total_points DESC) type_rnk
FROM pkmn
ORDER BY pokedex_number)

SELECT 
pokedex_number, name, type_1, type_2, total_points,
(avg_type1 + avg_type2)/2 as avg_avg,
rank() OVER (PARTITION BY ((avg_type1 + avg_type2)/2) ORDER BY total_points DESC) avg_rnk
FROM pkmn_win
```

__Top 15 Results:__

|pd_number|name                |type_1|type_2|total_points|avg_avg             |avg_rnk|
|-------|--------------------|------|------|------------|--------------------|-------|
|15     |Mega Beedrill       |Bug   |Poison|495         |393.8481156595191683|1      |
|545    |Scolipede           |Bug   |Poison|485         |393.8481156595191683|2      |
|49     |Venomoth            |Bug   |Poison|450         |393.8481156595191683|3      |
|168    |Ariados             |Bug   |Poison|400         |393.8481156595191683|4      |
|15     |Beedrill            |Bug   |Poison|395         |393.8481156595191683|5      |
|269    |Dustox              |Bug   |Poison|385         |393.8481156595191683|6      |
|544    |Whirlipede          |Bug   |Poison|360         |393.8481156595191683|7      |
|48     |Venonat             |Bug   |Poison|305         |393.8481156595191683|8      |
|543    |Venipede            |Bug   |Poison|260         |393.8481156595191683|9      |
|167    |Spinarak            |Bug   |Poison|250         |393.8481156595191683|10     |
|14     |Kakuna              |Bug   |Poison|205         |393.8481156595191683|11     |
|13     |Weedle              |Bug   |Poison|195         |393.8481156595191683|12     |
|127    |Pinsir              |Bug   |None  |500         |397.6944444444444445|1      |
|617    |Accelgor            |Bug   |None  |495         |397.6944444444444445|2      |
|314    |Illumise            |Bug   |None  |430         |397.6944444444444445|3      |


## Finding Total Count of Pokémon with a Unique Type Combination
There are 18 Pokémon types; however, the second typing (type 2) can technically have 19 unique value, due to no secondary typing requirements. For this reason, types were concatenated in their 'forward' position (type 1, type 2) and their 'backwards' position (type 2, type 1), and then counted in separate CTEs. The results were joined on the backwards concatenation to ensure all type combinations were present. The separate count columns were then summed to get the overall count.


```sql
with bkwd as (
select
count(pokedex_number) count_bkw,
concat(type_2,',',type_1) as bkw
from pkmn
group by bkw
),

fwrd as (select 
count(pokedex_number) count_fwd,
concat(type_1,',',type_2) as fwd
from pkmn 
group by fwd
)

select 
bkwd.bkw as grp,
bkwd.count_bkw as count_bkw,
(case when(fwrd.count_fwd is null) then 0 else fwrd.count_fwd end) as count_fwd,
(bkwd.count_bkw + (case when(fwrd.count_fwd is null) then 0 else fwrd.count_fwd end)) as ttl
from bkwd 
left outer join fwrd on fwrd.fwd = bkwd.bkw 
order by ttl desc
```

__Top 15 Results:__

|grp             |count_bkw|count_fwd|ttl|
|----------------|---------|---------|---|
|None,Water      |72       |0        |72 |
|None,Normal     |71       |0        |71 |
|None,Psychic    |44       |0        |44 |
|None,Grass      |43       |0        |43 |
|None,Fire       |34       |0        |34 |
|None,Electric   |33       |0        |33 |
|None,Fighting   |28       |0        |28 |
|Flying,Normal   |27       |0        |27 |
|None,Bug        |19       |0        |19 |
|None,Ice        |19       |0        |19 |
|None,Fairy      |19       |0        |19 |
|None,Ground     |17       |0        |17 |
|None,Rock       |16       |0        |16 |
|None,Poison     |16       |0        |16 |
|Poison,Grass    |15       |0        |15 |
|Flying,Bug      |14       |0        |14 |


### Returning List of Pokémon with a Unique Type Combination
To select the Pokémon with the most common typing, the previous query was put into another CTE and filtered in a WHERE clause against the complete data.

```sql
with bkwd as (
select
count(pokedex_number) count_bkw,
concat(type_2,',',type_1) as bkw
from pkmn
group by bkw
),

fwrd as (select 
count(pokedex_number) count_fwd,
concat(type_1,',',type_2) as fwd
from pkmn 
group by fwd
),

rnk as (select
bkwd.bkw as grp,
bkwd.count_bkw as count_bkw,
fwrd.count_fwd as count_fwd,
(bkwd.count_bkw + (case when(fwrd.count_fwd is null) then 0 else fwrd.count_fwd end)) as ttl,
rank () OVER(order by (bkwd.count_bkw + (case when(fwrd.count_fwd is null) then 0 else fwrd.count_fwd end)) desc) as rnk
from bkwd 
left outer join fwrd on fwrd.fwd = bkwd.bkw 
order by ttl desc)

select *,
rank () OVER(ORDER BY total_points desc) type1_rank
from pkmn 
where concat(type_2,',',type_1) = (select rnk.grp from rnk where rnk = 1)
```

__Top 15 Results:__

|pokedex_number|name                      |type_1|type_2|total_points|type1_rank|
|--------------|--------------------------|------|------|------------|----------|
|382           |Primal Kyogre             |Water |None  |770         |1         |
|382           |Kyogre                    |Water |None  |670         |2         |
|9             |Mega Blastoise            |Water |None  |630         |3         |
|746           |Wishiwashi School Form    |Water |None  |620         |4         |
|490           |Manaphy                   |Water |None  |600         |5         |
|245           |Suicune                   |Water |None  |580         |6         |
|350           |Milotic                   |Water |None  |540         |7         |
|818           |Inteleon                  |Water |None  |530         |8         |
|160           |Feraligatr                |Water |None  |530         |8         |
|9             |Blastoise                 |Water |None  |530         |8         |
|503           |Samurott                  |Water |None  |528         |11        |
|134           |Vaporeon                  |Water |None  |525         |12        |
|693           |Clawitzer                 |Water |None  |500         |13        |
|55            |Golduck                   |Water |None  |500         |13        |
|321           |Wailord                   |Water |None  |500         |13        |
