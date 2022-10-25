set -e
for f in $(find ttl_files_without_extra_P31/ -type f -name "*.ttl");
do
	echo $f
	/usr/java/java-11.0.5/bin/java -cp blazegraph.jar com.bigdata.rdf.store.DataLoader -verbose -namespace wd RWStore.properties $f
done
echo "DONE!"