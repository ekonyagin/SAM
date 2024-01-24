pip install gdown
mkdir pretrained_models
gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
gdown "https://drive.google.com/u/0/uc?id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0&export=download" -O pretrained_models/psp_ffhq_encode.pt
gdown "https://drive.google.com/u/0/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT&export=download" -O pretrained_models/stylegan2-ffhq-config-f.pt
gdown "https://drive.google.com/u/0/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn&export=download" -O pretrained_models/model_ir_se50.pth
gdown "https://drive.google.com/u/0/uc?id=1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh&export=download" -O pretrained_models/dex_age_classifier.pth