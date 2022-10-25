# Minimal setup
You can access a minimal setup from the following [link](https://uoe-my.sharepoint.com/:f:/g/personal/s1959796_ed_ac_uk/ErNL2lgTuI1Lu5vw8b6tEzEBk2YJ3QalLnYalG89e4Ge0g?e=KwkYC7) and run the following command.

```
bash start.sh
```

Alternatively, you can re-create the files by following the instructions below.


# Instructions to create triples file and load in blazegraph format

Server files including triples can be created by following the instructions below. We use blazegraph to host the server, however any other triple store should work too. 

## Create triples file
Get wikidata_proc_json_2 from [CSQA Dataset](https://amritasaha1812.github.io/CSQA/)
```
python json_to_triples.py
```

## Load triples to blazegraph
```
bash load_ttl.sh
```
This will produce wikidata.jnl

## Start server
- copy properties filem wikidata.jnl and wd_prefix.ttl into server_files
- Get the blazegraph.jar via [Blazegraph](https://blazegraph.com/)
```
export PATH=$PATH:$JAVA_HOME
cd path_to/server_files
/usr/java/java-11.0.5/bin/java -server -Xmx150g -XX:+UseG1GC -Dcom.bigdata.rdf.sail.sparql.PrefixDeclProcessor.additionalDeclsFile=wd_prefix.ttl -jar blazegraph.jar
```