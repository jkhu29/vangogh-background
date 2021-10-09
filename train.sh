set -aux
cd convert
python -W ignore train.py --batch_size 4 --niter 1 --batch_scale 4
cd ../segment
python -W ignore train.py --batch_size 16 --niter 1