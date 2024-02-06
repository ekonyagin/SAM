python scripts/inference.py \
--exp_dir=/home/e.konyagin/SAM/pretrained_psp \
--checkpoint_path=/home/e.konyagin/SAM/pretrained_psp/checkpoints/best_model.pt \
--data_path=/home/e.konyagin/test_images \
--test_batch_size=8 \
--test_workers=8 \
--target_age=20,25,30,35,40,45,50,55,60,65,70
