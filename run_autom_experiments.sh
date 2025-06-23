echo -e "Starting Experiments"
python3 -m experiments.scenapp_tests.benchmarks.eps_curves
echo -e "Risk Comparison Curve Generated"
echo -e "Spiral Reach" > results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.reach_spiral >> results/autom.txt
echo -e "\nSpiral Safe" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.barr_spiral >> results/autom.txt
echo -e \n"Spiral RWA" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.rwa_spiral >> results/autom.txt
echo -e "Spiral Benchmarks Complete"
echo -e "\nSpiral Partially Unsafe" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.barr_spiral_unsafe >> results/autom.txt
echo -e "Partially Unsafe Complete"

echo -e "\nSpiral Partially Unsafe" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.barr_spiral_unsafe >> results/autom.txt

echo -e "\n8D Barrier" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.high_ord_DT >> results/autom.txt

# DT Comparison with Nejati
echo -e "\nDC Motor Our Appraoch" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.DC_Motor >> results/autom.txt
echo -e "\nDC Motor Nejati" >> results/autom.txt
python3 -m Nejati23.DC >> results/autom.txt
echo -e "DC Motor Experiments Completed"

echo -e "\n4D Our Appraoch" >> results/autom.txt
python3 -m experiments.scenapp_tests.benchmarks.Barr4D_DT >> results/autom.txt
echo -e "\n4D Nejati" >> results/autom.txt
python3 -m Nejati23.HighD_DT >> results/autom.txt
