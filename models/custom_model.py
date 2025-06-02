# models/custom_model.py

import torch
import torch.nn as nn
import timm

class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, input_dim // 2,
            num_layers=2, batch_first=True,
            dropout=0.1, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, bags):
        # bags: (batch_size, num_slices, input_dim)
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)                     # (batch, slices, input_dim)
        attn_scores = self.attention(bags_lstm).squeeze(-1) # (batch, slices)
        attn_weights = torch.softmax(attn_scores, dim=-1)   # (batch, slices)
        weighted_instances = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, slices)
            bags_lstm                   # (batch, slices, input_dim)
        ).squeeze(1)                   # (batch, input_dim)
        return weighted_instances       # (batch, input_dim)

class ConvNextLSTM(nn.Module):
    def __init__(self, pretrained=False, in_chans=1, class_num=1):
        super(ConvNextLSTM, self).__init__()
        # Load the ConvNeXt‐small backbone (no classifier head, just features)
        self.backbone = 'convnext_small.fb_in22k_ft_in1k_384'
        backbone = timm.create_model(
            self.backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool='',
            num_classes=0
        )
        self.encoder = backbone
        num_features = self.encoder.num_features  # e.g. 768 for ConvNeXt‐small

        # Pool and flatten per‐slice features
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )

        # LSTM + attention on the sequence of slice‐features
        self.lstm = LSTMMIL(num_features)

        # Final head to turn that bag‐embedding into logits
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, class_num),  # outputs [batch, class_num]
        )

    def forward(self, x):
        # x shape: (batch, in_chans, num_slices, H, W)
        bs, in_chans, n_slices, H, W = x.shape

        # Reorder so that each slice is passed through ConvNeXt
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, in_chans, H, W)            # (batch * slices, in_chans, H, W)

        # Extract per‐slice features
        x = self.encoder.forward_features(x)       # (batch*slices, num_features, ?, ?)
        x = self.flatten(x)                        # (batch*slices, num_features)

        # Reshape back into (batch, slices, num_features)
        x = x.view(bs, n_slices, -1)               # (batch, slices, num_features)

        # LSTM + attention → (batch, num_features)
        x = self.lstm(x)

        # Final classification head → (batch, class_num)
        x = self.head(x)
        return x
