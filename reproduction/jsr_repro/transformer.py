"""Independent seven-frame late-fusion Transformer inspired by supplied speech.

No original Transformer source/weights/layout was publicly verified. This
model's attention, amplitude, pairing and upsampler equations are assumptions.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

IMPLEMENTATION = "speech-inspired-transformer-v1"
FIELDS = ("raw", "shifts", "exposure", "transmission", "variance", "saturation", "valid", "black_invalid")


def pack(raw):
    return torch.cat([raw[..., 0::2, 0::2], raw[..., 0::2, 1::2], raw[..., 1::2, 0::2], raw[..., 1::2, 1::2]], -3)


def validate_inputs(sample):
    if any(key not in sample for key in FIELDS):
        raise ValueError("Transformer requires raw, shifts, exposure, transmission, variance, saturation, valid, black_invalid")
    raw = sample["raw"]
    if raw.ndim != 5 or raw.shape[1:3] != (7, 1) or min(raw.shape[-2:]) < 16 or any(d % 2 for d in raw.shape[-2:]):
        raise ValueError("Transformer RAW must be[B,7,1,H,W], even dimensions>=16")
    b = raw.shape[0]
    if 'capture_order' in sample:
        ranks=sample['capture_order']
        if ranks.shape!=(b,7) or not torch.isfinite(ranks).all() or not torch.equal(ranks.sort(1).values,torch.arange(7,device=ranks.device).expand(b,-1).to(ranks)):
            raise ValueError('capture_order must be a permutation of temporal ranks 0..6')
    if sample["shifts"].shape != (b, 7, 2) or sample["exposure"].shape != (b, 7) or sample["transmission"].shape != (b, 3):
        raise ValueError("metadata shapes: shifts[B,7,2], exposure[B,7], transmission[B,3]")
    for key in FIELDS:
        if not torch.isfinite(sample[key]).all():
            raise ValueError(f"nonfinite Transformer metadata: {key}")
    if (sample["shifts"][:, 0] != 0).any() or (sample["exposure"] <= 0).any() or (sample["exposure"][:, 0] != 1).any() or (sample["transmission"] <= 0).any() or (sample["transmission"] > 1).any():
        raise ValueError("reference shifts=0, reference exposure=1, positive exposure and transmission<=1 required")
    for key in ("variance", "saturation", "valid", "black_invalid"):
        if sample[key].shape != raw.shape or (sample[key] < 0).any():
            raise ValueError(f"{key} must match RAW and be nonnegative")
    for key in ("saturation", "valid", "black_invalid"):
        if (sample[key] > 1).any():
            raise ValueError(f"{key} must be in[0,1]")


def align_planes(raw, variance, saturation, valid, black, shifts, geometry=None):
    """Inverse translation dx/2, squared bilinear coefficients for variance.

    CFA phase cancels between same-color source/reference packed planes.
    Masks are conservative over nonzero contributors; padding has no evidence.
    Independent input-noise assumption; interpolation correlations not modeled.
    """
    n, k, c, h, w = raw.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=raw.device, dtype=raw.dtype), torch.arange(w, device=raw.device, dtype=raw.dtype), indexing="ij")
    sx,sy = (shifts[...,0],shifts[...,1]) if shifts.ndim==5 else (shifts[...,0,None,None],shifts[...,1,None,None])
    x = xx[None,None] - sx / 2
    y = yy[None,None] - sy / 2
    if geometry is not None:
        results=[]
        for channel,(cx,cy) in enumerate(((0,0),(1,0),(0,1),(1,1))):
            native=geometry.inverse[:,:,cy::2,cx::2]
            # Recursive one-channel sampler receives dense displacement in packed units.
            displacement=torch.stack((2*(xx[None,None]-(native[...,0]-cx)/2),2*(yy[None,None]-(native[...,1]-cy)/2)),-1)
            results.append(align_planes(*(v[:,:,channel:channel+1] for v in (raw,variance,saturation,valid,black)), displacement))
        return tuple(torch.cat([r[i] for r in results],2) for i in range(5))
    x0, y0 = x.floor(), y.floor()
    wx, wy = x - x0, y - y0
    out, var = torch.zeros_like(raw), torch.zeros_like(variance)
    sat, blk, good = torch.zeros_like(saturation), torch.zeros_like(black), torch.ones_like(valid)
    for dx, dy, weight in ((0, 0, (1-wx)*(1-wy)), (1, 0, wx*(1-wy)), (0, 1, (1-wx)*wy), (1, 1, wx*wy)):
        xi, yi = x0 + dx, y0 + dy
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        index = (yi.clamp(0, h-1).long() * w + xi.clamp(0, w-1).long()).reshape(n, k, 1, h*w).expand(-1, -1, c, -1)
        values = [torch.gather(v.flatten(-2), -1, index).reshape(n,k,c,h,w) for v in (raw, variance, saturation, valid, black)]
        weight = weight[:, :, None]
        active = weight > 0
        out += values[0] * weight
        var += values[1] * weight.square()
        sat = torch.maximum(sat, torch.where(active, values[2], torch.zeros_like(sat)))
        blk = torch.maximum(blk, torch.where(active, values[4], torch.zeros_like(blk)))
        good *= torch.where(active, values[3] * inside[:, :, None], torch.ones_like(good))
    return out, var, sat, good, blk


def edge_amplitude(merged):
    """3x3 edge-weighted RMS. Edge distances are scale-free, no signal epsilon.

    Positive scaling of merged scales amplitude identically; zero signal uses
    a safe denominator1 and amplitude0. This is not the author's exact filter.
    """
    b, c, h, w = merged.shape
    neighbors = F.unfold(F.pad(merged, (1,1,1,1), mode="replicate"), 3).reshape(b,c,9,h,w)
    center = merged[:, :, None]
    energy = merged.square().mean(1, keepdim=True)
    reference = F.avg_pool2d(F.pad(energy, (1,1,1,1), mode="replicate"), 3, stride=1)
    denominator = torch.where(reference > 0, reference, torch.ones_like(reference))
    distance = (neighbors-center).square().mean(1) / denominator
    weight = torch.exp(-distance * 4)
    local_energy = (neighbors.square().mean(1) * weight).sum(1,keepdim=True) / weight.sum(1,keepdim=True)
    # Safe sqrt at0: direct sqrt's infinite derivative would yield NaN grads.
    return torch.where(local_energy > 0, torch.sqrt(torch.where(local_energy > 0, local_energy, torch.ones_like(local_energy))), torch.zeros_like(local_energy))


def planes_to_rgb(planes):
    """CFA-coordinate-aware bilinear baseline, packed->2x native (4x packed)."""
    b, _, h, w = planes.shape
    oh, ow = h*4, w*4
    yy, xx = torch.meshgrid(torch.arange(oh, device=planes.device, dtype=planes.dtype), torch.arange(ow, device=planes.device, dtype=planes.dtype), indexing="ij")
    native_x, native_y = (xx+.5)/2-.5, (yy+.5)/2-.5
    channels = []
    for i, (px,py) in enumerate(((0,0),(1,0),(0,1),(1,1))):
        grid = torch.stack((2*((native_x-px)/2+.5)/w-1, 2*((native_y-py)/2+.5)/h-1), -1)
        channels.append(F.grid_sample(planes[:,i:i+1], grid[None].expand(b,-1,-1,-1), padding_mode="border", align_corners=False))
    return torch.cat((channels[0], (channels[1]+channels[2])/2, channels[3]), 1)


class ExchangeBlock(nn.Module):
    def __init__(self, width, heads, window):
        super().__init__()
        self.window = window
        self.spatial = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.temporal = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.norm_s, self.norm_t = nn.LayerNorm(width), nn.LayerNorm(width)
        self.difference = nn.Conv2d(width, width, 3, padding=1)
        self.ffn = nn.Sequential(nn.Conv2d(width,width*2,1), nn.GELU(), nn.Conv2d(width*2,width,1))

    def forward(self, x):
        b,k,c,h,w = x.shape
        win = self.window
        flat = x.reshape(b*k,c,h,w)
        ph,pw = (-h)%win, (-w)%win
        padded = F.pad(flat,(0,pw,0,ph),mode="replicate")
        hh,ww = padded.shape[-2:]
        tokens = padded.reshape(b*k,c,hh//win,win,ww//win,win).permute(0,2,4,3,5,1).reshape(-1,win*win,c)
        normalized = self.norm_s(tokens)
        tokens = tokens + self.spatial(normalized,normalized,normalized,need_weights=False)[0]
        flat = tokens.reshape(b*k,hh//win,ww//win,win,win,c).permute(0,5,1,3,2,4).reshape(b*k,c,hh,ww)[...,:h,:w]
        x = flat.reshape(b,k,c,h,w)
        tokens = x.permute(0,3,4,1,2).reshape(b*h*w,k,c)
        normalized = self.norm_t(tokens)
        tokens = tokens + self.temporal(normalized,normalized,normalized,need_weights=False)[0]
        x = tokens.reshape(b,h,w,k,c).permute(0,3,4,1,2)
        difference = (x - x.roll(1,1)).reshape(b*k,c,h,w)
        flat = x.reshape(b*k,c,h,w) + self.difference(difference)
        return (flat+self.ffn(flat)).reshape(b,k,c,h,w)


class PairReduce(nn.Module):
    def __init__(self, width, rotation):
        super().__init__()
        self.rotation = rotation
        self.gate = nn.Conv2d(width*2,width,1)
        self.mix = nn.Conv2d(width*2,width,3,padding=1)

    def forward(self, x):
        x = x.roll(self.rotation,1)
        output = []
        for i in range(0,x.shape[1]-1,2):
            a,b = x[:,i],x[:,i+1]
            pair = torch.cat((a,b),1)
            gate = torch.sigmoid(self.gate(pair))
            output.append(gate*a+(1-gate)*b+.05*self.mix(torch.cat((a-b,(a+b)/2),1)))
        if x.shape[1]%2:
            output.append(x[:,-1])
        return torch.stack(output,1)


class SpeechTransformer(nn.Module):
    def __init__(self, width=16, heads=2, window=4, scale=2, frames=7, capture_order_mode="legacy-storage-v1"):
        super().__init__()
        if frames != 7 or scale != 2 or width < 4 or heads < 1 or width%heads or not 2<=window<=8:
            raise ValueError("model requires K7, scale2, width>=4 divisible by heads, window2..8")
        if capture_order_mode not in ('legacy-storage-v1','explicit-ranks-v1'):
            raise ValueError('unknown capture order semantics')
        self.capture_order_mode = capture_order_mode
        self.embed = nn.Conv2d(21,width,3,padding=1)
        self.enc = nn.ModuleList([ExchangeBlock(width,heads,window) for _ in range(3)])
        self.down = nn.ModuleList([nn.Conv2d(width,width,3,stride=2,padding=1) for _ in range(2)])
        self.skip = nn.ModuleList([nn.Conv2d(width*2,width,3,padding=1) for _ in range(2)])
        self.dec = nn.ModuleList([ExchangeBlock(width,heads,window) for _ in range(2)])
        self.pairs = nn.ModuleList([PairReduce(width,i+1) for i in range(3)])
        self.upsample = nn.Sequential(nn.Conv2d(width,width*4,3,padding=1), nn.PixelShuffle(2), nn.GELU(),
                                      nn.Conv2d(width,width*4,3,padding=1), nn.PixelShuffle(2), nn.GELU())
        self.rgb_projection = nn.Conv2d(width,3,1)
        with torch.no_grad():
            self.rgb_projection.weight.mul_(.01)
            self.rgb_projection.bias.zero_()

    def forward(self, sample, trace=False):
        validate_inputs(sample)
        if "geometry" in sample:
            from .geometry import validate_geometry
            validate_geometry(sample["geometry"], sample["raw"])
        raw = pack(sample["raw"])
        b,k,c,h,w = raw.shape
        values = align_planes(raw, pack(sample["variance"]), pack(sample["saturation"]), pack(sample["valid"]), pack(sample["black_invalid"]), sample["shifts"], sample.get("geometry"))
        observed,var,sat,valid,black = values
        transmission = sample["transmission"][:,[0,1,1,2]][:,None,:,None,None]
        exposure = sample["exposure"][:,:,None,None,None]
        factor = transmission*exposure
        radiance,variance = observed/factor,var/factor.square()
        reliability = valid*(1-sat)
        count = reliability.sum(1)
        fallback = (radiance*valid).sum(1)/valid.sum(1).clamp_min(1)
        merged = torch.where(count>0,(radiance*reliability).sum(1)/count.clamp_min(1),fallback)
        amplitude = edge_amplitude(merged)
        divisor = torch.where(amplitude>0,amplitude,torch.ones_like(amplitude))[:,None]
        ev = torch.log2(exposure).expand(b,k,1,h,w)
        feature = torch.cat((radiance/divisor,variance/divisor.square(),sat,valid,black,ev),2)
        if self.capture_order_mode == 'explicit-ranks-v1':
            if 'capture_order' not in sample:
                raise ValueError('explicit-ranks-v1 requires capture_order metadata')
            order=sample['capture_order'].argsort(1)
            feature=feature.gather(1,order[:,:,None,None,None].expand_as(feature))
        x = self.embed(feature.reshape(b*k,21,h,w)).reshape(b,k,-1,h,w)
        shapes = {"embedding": list(x.shape)}
        skips = []
        for i,block in enumerate(self.enc):
            x = block(x)
            shapes[f"encoder{i}"] = list(x.shape)
            if i<2:
                skips.append(x)
                x = self.down[i](x.flatten(0,1)).reshape(b,k,x.shape[2],(x.shape[-2]+1)//2,(x.shape[-1]+1)//2)
        for i,(conv,block,skip) in enumerate(zip(self.skip,self.dec,reversed(skips))):
            flat = F.interpolate(x.flatten(0,1),size=skip.shape[-2:],mode="bilinear",align_corners=False)
            x = conv(torch.cat((flat,skip.flatten(0,1)),1)).reshape(b,k,-1,*skip.shape[-2:])
            x = block(x)
            shapes[f"decoder{i}"] = list(x.shape)
        for i,pair in enumerate(self.pairs):
            x = pair(x)
            shapes[f"late_pair{i}"] = list(x.shape)
        hidden = self.upsample(x[:,0])
        shapes["shared_phase_latent"] = list(hidden.shape)
        correction = self.rgb_projection(hidden)
        baseline = planes_to_rgb(merged)
        amplitude_hr = F.interpolate(amplitude,size=baseline.shape[-2:],mode="bilinear",align_corners=False)
        output = baseline + correction*amplitude_hr
        output = torch.where(amplitude_hr>0,output,torch.zeros_like(output))
        result = {"rgb": output, "baseline": baseline, "amplitude": amplitude_hr,
                  "all_saturated_or_invalid_packed": (count==0).float()}
        if trace:
            result["trace"] = shapes
        return result
