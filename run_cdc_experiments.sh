echo "Starting experiments"

echo "Our Approach for Jet Engine\n" > results/cdc_results.txt
python3 -m experiments.scenapp_tests.benchmarks.jet_engine >> results/cdc_results.txt
echo "\n\n Nejati23 Results for Jet Engine\n" >> results/cdc_results.txt
python3 -m Nejati23.jet >> results/cdc_results.txt

echo "\n\nOur Approach for 4D Model\n" >> results/cdc_results.txt
python3 -m experiments.scenapp_tests.benchmarks.Barr4D >> results/cdc_results.txt
echo "\n\n Nejati23 Results for 4D\n" >> results/cdc_results.txt
python3 -m Nejati23.HighD >> results/cdc_results.txt

echo "4D Tests Complete"
echo "\n\nOur Approach for unsafe Model\n" >> results/cdc_results.txt
python3 -m experiments.scenapp_tests.benchmarks.jet_engine_unsafe >> results/cdc_results.txt
