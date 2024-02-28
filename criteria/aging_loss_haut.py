import torch
import torch.nn.functional as F
from torch import nn

from configs.paths_config import model_paths
from models.convnext_age import RegressionModel


class AgingLoss(nn.Module):

    def __init__(self, opts):
        super(AgingLoss, self).__init__()
        self.age_net = RegressionModel(model_name="convnext_pico", num_classes=100)
        self, age_net.load_state_dict(
            torch.load("pretrained_models/age_model_cn_pico.pt", map_location="cpu")
        )
        self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100
        self.opts = opts

    def __get_predicted_age(self, age_pb):
        return age_pb.argmax(1).astype(torch.float32)

    def extract_ages(self, x):
        x = F.interpolate(x, size=(320, 320), mode="bilinear")
        predict_age_pb = self.age_net(x)  # BS x 100 classes
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age

    def forward(self, y_hat, y, target_ages, id_logs, label=None):

        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []

        input_ages = self.extract_ages(y) / 100.0
        with torch.no_grad():
            y_hat -= torch.tensor([0.485, 0.456, 0.406])[:, None, None][None, ...]
            y_hat /= torch.tensor([0.229, 0.224, 0.225])[:, None, None][None, ...]
        output_ages = self.extract_ages(y_hat) / 100.0

        for i in range(n_samples):
            # if id logs for the same exists, update the dictionary
            if len(id_logs) > i:
                id_logs[i].update(
                    {
                        f"input_age_{label}": float(input_ages[i]) * 100,
                        f"output_age_{label}": float(output_ages[i]) * 100,
                        f"target_age_{label}": float(target_ages[i]) * 100,
                    }
                )
            # otherwise, create a new entry for the sample
            else:
                id_logs.append(
                    {
                        f"input_age_{label}": float(input_ages[i]) * 100,
                        f"output_age_{label}": float(output_ages[i]) * 100,
                        f"target_age_{label}": float(target_ages[i]) * 100,
                    }
                )

        loss = F.mse_loss(output_ages, target_ages)
        return loss, id_logs
