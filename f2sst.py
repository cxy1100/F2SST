import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedSpectralAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(GatedSpectralAttention, self).__init__()
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int((N - 1) ** 0.5)
        cls_token = x[:, 0:1, :]
        patch_tokens = x[:, 1:, :]

        x_img = patch_tokens.transpose(1, 2).view(B, C, H, W)
        x_fft = torch.fft.fft2(x_img, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft)
        x_real = x_fft.real

        freq_summary = F.adaptive_avg_pool2d(x_real, 1).view(B, C)
        gates = self.fc(freq_summary).view(B, C, 1, 1)

        x_filtered = x_real * gates
        x_filtered = F.adaptive_avg_pool2d(x_filtered, (H, W))

        x_out = x_filtered.flatten(2).transpose(1, 2)
        x_out = torch.cat([cls_token, x_out], dim=1)
        return x_out


class F2SST(nn.Module):
    def __init__(self, in_dim=384, freq_strategy='all'):
        super(F2SST, self).__init__()
        self.freq_strategy = freq_strategy

        self.fc1 = Mlp(in_features=in_dim, hidden_features=in_dim // 4, out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)
        self.fc2 = Mlp(in_features=196 ** 2, hidden_features=256, out_features=1)

        self.freq_attention = GatedSpectralAttention(dim=in_dim)

    def fft_process(self, x):
        B, N, C = x.shape
        H = W = int((N - 1) ** 0.5)
        cls_token = x[:, 0:1, :]
        x = x[:, 1:, :].permute(0, 2, 1).contiguous().view(B, C, H, W)
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft)
        x_real = x_fft.real

        if self.freq_strategy != 'all':
            mask = torch.zeros_like(x_real)
            cy, cx = H // 2, W // 2
            if self.freq_strategy == 'low':
                r = min(H, W) // 4
                mask[:, :, cy - r:cy + r, cx - r:cx + r] = 1
            elif self.freq_strategy == 'high':
                mask[:] = 1
                r = min(H, W) // 4
                mask[:, :, cy - r:cy + r, cx - r:cx + r] = 0
            elif self.freq_strategy == 'center_block':
                r = min(H, W) // 8
                mask[:, :, cy - r:cy + r, cx - r:cx + r] = 1
            elif self.freq_strategy == 'band':
                r1 = min(H, W) // 6
                r2 = min(H, W) // 3
                yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                dist = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt().to(x_real.device)
                mask_band = ((dist >= r1) & (dist <= r2)).float()
                mask = mask_band.unsqueeze(0).unsqueeze(0).expand_as(x_real)
            x_real = x_real * mask

        x_reduced = F.adaptive_avg_pool2d(x_real, (H, W))
        x_flat = x_reduced.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, x_flat], dim=1)

    def forward(self, feat_query, feat_shot, args):
        B, N, C = feat_query.shape
        H = W = int((N - 1) ** 0.5)

        feat_query = self.fft_process(feat_query)
        feat_shot = self.fft_process(feat_shot)

        feat_query = self.freq_attention(feat_query)
        feat_shot = self.freq_attention(feat_shot)

        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query
        feat_shot = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot

        feat_query = self.fc_norm1(feat_query)
        feat_shot = self.fc_norm1(feat_shot)

        query_class = feat_query[:, 0, :].unsqueeze(1)
        query_image = feat_query[:, 1:, :]
        support_class = feat_shot[:, 0, :].unsqueeze(1)
        support_image = feat_shot[:, 1:, :]

        feat_query = query_image + 1.0 * query_class
        feat_shot = support_image + 1.0 * support_class

        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)

        feat_shot = feat_shot.contiguous().reshape(args.shot, -1, N - 1, C).mean(dim=0)
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)

        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx].unsqueeze(0)
            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2))
            out = out.flatten(1)
            out = self.fc2(out.pow(2))
            out = out.transpose(0, 1)
            results.append(out)

        return results, None
