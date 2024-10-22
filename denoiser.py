import torch
from torch import nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        # Create sinusoidal position embeddings (same as those from the transformer)
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class LatentEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels

    def forward(self, z, drop_mask):
        """
        * `z` is the latent code
        * `drop_mask`: mask out the condition if drop_mask == 1
        """
        drop_mask = drop_mask[:, None]
        drop_mask = drop_mask.repeat(1, self.n_channels)
        drop_mask = 1 - drop_mask  # need to flip 0 <-> 1
        z = z * drop_mask
        return z


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k

        self.norm = nn.GroupNorm(32, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = 1 / math.sqrt(math.sqrt(d_k))
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """
        batch_size, n_channels, height, width = x.shape
        # Normalize and rearrange to `[batch_size, seq, n_channels]`
        h = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)

        # {q, k, v} all have a shape of `[batch_size, seq, n_heads, d_k]`
        qkv = self.projection(h).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q * self.scale, k * self.scale) # More stable with f16 than dividing afterwards
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape to `[batch_size, seq, n_heads * d_k]` and transform to `[batch_size, seq, n_channels]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res + x


class Upsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            return self.conv(x)
        else:
            return x


class Downsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, z_channels, dropout=0.1, up=False, down=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `z_channels` is the number channels in the latent code derived by the resnet encoder
        * `dropout` is the dropout rate
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels) 
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for embeddings
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, 2 * out_channels)
        )
        self.z_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_channels, 2 * out_channels)
        )

        # BigGAN style: use resblock for up/downsampling
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, use_conv=False)
            self.x_upd = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

    def forward(self, x, t, z):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        * `z` has shape `[batch_size, z_channels]`
        """
        if self.updown:
            h = self.norm2(self.conv1(self.h_upd(self.act1(self.norm1(x)))))
            x = self.x_upd(x)
        else:
            h = self.norm2(self.conv1(self.act1(self.norm1(x))))

        # Adaptive Group Normalization
        t_s, t_b = self.time_emb(t).chunk(2, dim=1)
        z_s, z_b = self.z_emb(z).chunk(2, dim=1)
        h = t_s[:, :, None, None] * h + t_b[:, :, None, None]
        h = z_s[:, :, None, None] * h + z_b[:, :, None, None]

        h = self.conv2(self.act2(h))
        return h + self.shortcut(x)


class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, z_channels, has_attn, attn_channels_per_head, dropout):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, z_channels, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t, z):
        x = self.res(x, t, z)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, z_channels, attn_channels_per_head, dropout):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout)
        self.attn = AttentionBlock(n_channels, attn_channels_per_head)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout)

    def forward(self, x, t, z):
        x = self.res1(x, t, z)
        x = self.attn(x)
        x = self.res2(x, t, z)
        return x


class UpsampleRes(nn.Module):
    def __init__(self, n_channels, time_channels, z_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout, up=True)

    def forward(self, x, t, z):
        return self.op(x, t, z)


class DownsampleRes(nn.Module):
    def __init__(self, n_channels, time_channels, z_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout, down=True)

    def forward(self, x, t, z):
        return self.op(x, t, z)
 

class UNet_decoder(nn.Module):
    def __init__(self, image_shape = [3, 32, 32], n_channels = 128,
                 ch_mults = (1, 2, 2, 2),
                 is_attn = (False, True, False, False),
                 attn_channels_per_head = None,
                 dropout = 0.1,
                 n_blocks = 2,
                 use_res_for_updown = False,
                 z_channels = 128):
        """
        * `image_shape` is the (channel, height, width) size of images.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `n_channels * ch_mults[i]`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `dropout` is the dropout rate
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        * `use_res_for_updown` indicates whether to use ResBlocks for up/down sampling (BigGAN-style)
        * `z_channels` is the number channels in the latent code derived by the resnet encoder
        """
        super().__init__()

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(image_shape[0], n_channels, kernel_size=3, padding=1)

        # Time embedding layer.
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding(time_channels)

        # Latent embedding layer.
        self.z_emb = LatentEmbedding(z_channels)

        # Down stages
        down = []
        in_channels = n_channels
        h_channels = [n_channels]
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks` at the same resolution
            down.append(ResAttBlock(in_channels, out_channels, time_channels, z_channels, is_attn[i], attn_channels_per_head, dropout))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 1):
                down.append(ResAttBlock(out_channels, out_channels, time_channels, z_channels, is_attn[i], attn_channels_per_head, dropout))
                h_channels.append(out_channels)
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                if use_res_for_updown:
                    down.append(DownsampleRes(out_channels, time_channels, z_channels, dropout))
                else:
                    down.append(Downsample(out_channels))
                h_channels.append(out_channels)
            in_channels = out_channels
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, z_channels, attn_channels_per_head, dropout)

        # Up stages
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks + 1` at the same resolution
            for _ in range(n_blocks + 1):
                up.append(ResAttBlock(in_channels + h_channels.pop(), out_channels, time_channels, z_channels, is_attn[i], attn_channels_per_head, dropout))
                in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                if use_res_for_updown:
                    up.append(UpsampleRes(out_channels, time_channels, z_channels, dropout))
                else:
                    up.append(Upsample(out_channels))
        assert not h_channels
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(out_channels, image_shape[0], kernel_size=3, padding=1)

    def forward(self, x, t, z, drop_mask, ret_activation=False):
        if not ret_activation:
            return self.forward_core(x, t, z, drop_mask)

        activation = {}
        def namedHook(name):
            def hook(module, input, output):
                activation[name] = output
            return hook
        hooks = {}
        no = 0
        for blk in self.up:
            if isinstance(blk, ResAttBlock):
                no += 1
                name = f'out_{no}'
                hooks[name] = blk.register_forward_hook(namedHook(name))

        result = self.forward_core(x, t, z, drop_mask)
        for name in hooks:
            hooks[name].remove()
        return result, activation

    def forward_core(self, x, t, z, drop_mask):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        * `z` has shape `[batch_size, z_channels]`
        * `drop_mask` has shape `[batch_size]`
        """

        t = self.time_emb(t)
        x = self.image_proj(x)
        z = self.z_emb(z, drop_mask)

        # `h` will store outputs at each resolution for skip connection
        h = [x]

        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            elif isinstance(m, DownsampleRes):
                x = m(x, t, z)
            else:
                x = m(x, t, z).contiguous()
            h.append(x)

        x = self.middle(x, t, z).contiguous()

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            elif isinstance(m, UpsampleRes):
                x = m(x, t, z)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, z).contiguous()

        return self.final(self.act(self.norm(x)))

if __name__ == "__main__":
    device = "cuda"
    net = UNet_decoder(image_shape=(3, 64, 64)).to(device)
    x = torch.rand(8, 3, 64, 64).to(device)
    t = torch.randint(0, 10, (8,)).to(device)
    z = torch.rand(8, 128).to(device)
    drop_mask = torch.randint(0,1, (8,)).to(device)
    print(net(x,t,z,drop_mask).shape)
