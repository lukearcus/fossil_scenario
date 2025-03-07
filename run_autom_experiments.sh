echo "Starting Experiments"
python3 -m experiments.scenapp_tests.eps_curves
echo "Risk Comparison Curve Generated"
echo "Spiral Reach" > results/autom.txt
python3 -m experiments.scenapp_tests.reach_spiral >> results/autom.txt
echo "\nSpiral Safe" >> results/autom.txt
python3 -m experiments.scenapp_tests.barr_spiral >> results/autom.txt
echo "\nSpiral RWA" >> results/autom.txt
python3 -m experiments.scenapp_tests.rwa_spiral >> results/autom.txt
echo "Spiral Benchmarks Complete"
echo "\nSpiral Partially Unsafe" >> results/autom.txt
python3 -m experiments.scenapp_tests.barr_spiral_unsafe >> results/autom.txt
echo "Partially Unsafe Complete"

# DT Comparison with Nejati
echo "\nDC Motor Our Appraoch" >> results/autom.txt
python3 -m experiments.scenapp_tests.DC_Motor >> results/autom.txt
echo "\nDC Motor Nejati" >> results/autom.txt
python3 -m Nejati23.DC >> results/autom.txt
echo "DC Motor Experiments Completed"

echo "\n4D Our Appraoch" >> results/autom.txt
python3 -m experiments.scenapp_tests.Barr4D_DT >> results/autom.txt
echo "\n4D Nejati" >> results/autom.txt
python3 -m Nejati23.highD_DT >> results/autom.txt
