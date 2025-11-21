
CUDA_VISIBLE_DEVICES=3 python main.py --gpu 0 --way 5 --test_way 5 --shot 5 --exp f2sst-5way-5shot --use_fft True --use_attention True --use_fusion True > f2sst-5way-5shot.txt &
CUDA_VISIBLE_DEVICES=5 python main.py --gpu 0 --way 5 --test_way 5 --shot 1 --exp f2sst-5way-1shot --use_fft True --use_attention True --use_fusion True > f2sst-5way-1shot.txt &

