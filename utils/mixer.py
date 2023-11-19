import torch
import numpy as np

class Mixer:
    def mix(self, a, b, *args):
        """
        a, b: FloatTensor or ndarray
        return: same type and shape as a
        """
        pass
        
class HalfMixer(Mixer):
    def __init__(self, channel_first=True, vertical=None, gap=0, jitter=3, shake=True):
        self.channel_first = channel_first
        self.vertical = vertical
        self.gap = gap
        self.jitter = jitter
        self.shake = shake

    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        gap = round(self.gap / 2)
        jitter = np.random.randint(-self.jitter, self.jitter + 1)

        pivot = np.random.randint(0, h // 2 - jitter) if self.shake else h // 4 - jitter // 2
        a_b[:, :h // 2 + jitter - gap, :] = a[:, pivot:pivot + h // 2 + jitter - gap, :]
        pivot = np.random.randint(-jitter, h // 2) if self.shake else h // 4 - jitter // 2
        a_b[:, h // 2 + jitter + gap:, :] = b[:, pivot + jitter + gap:pivot + h // 2, :]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b

                        
class CropPasteMixer(Mixer):
    def __init__(self, channel_first=True, max_overlap=0.15, max_iter=30, resize=(0.5, 2), shift=0.3):
        self.channel_first = channel_first
        self.max_overlap = max_overlap
        self.max_iter = max_iter
        self.resize = resize
        self.shift = shift
        
    def get_overlap(self, bboxA, bboxB):
        x1a, y1a, x2a, y2a = bboxA
        x1b, y1b, x2b, y2b = bboxB

        left = max(x1a, x1b)
        right = min(x2a, x2b)
        bottom = max(y1a, y1b)
        top = min(y2a, y2b)

        if left < right and bottom < top:
            areaA = (x2a - x1a) * (y2a - y1a)
            areaB = (x2b - x1b) * (y2b - y1b)
            return (right - left) * (top - bottom) / min(areaA, areaB)
        return 0

    def stamp(self, a, b, bboxA, max_overlap, max_iter):
        _, Ha, Wa = a.shape
        _, Hb, Wb = b.shape
        assert Ha > Hb and Wa > Wb

        best_overlap = 999
        best_bboxB = None
        overlap_inc = max_overlap / max_iter
        max_overlap = 0

        for _ in range(max_iter):
            cx = np.random.randint(0, Wa - Wb)
            cy = np.random.randint(0, Ha - Hb)
            bboxB = (cx, cy, cx + Wb, cy + Hb)
            overlap = self.get_overlap(bboxA, bboxB)

            if best_overlap > overlap:
                best_overlap = overlap
                best_bboxB = bboxB
            else:
                overlap = best_overlap

            # print(overlap, max_overlap)

            # check the threshold
            if overlap <= max_overlap:
                break
            max_overlap += overlap_inc

        cx, cy = best_bboxB[:2]
        a_b = a.clone()
        a_b[:, cy:cy + Hb, cx:cx + Wb] = b[:]
        return a_b, best_overlap

    def crop_bbox(self, image, bbox):
        x1, y1, x2, y2 = bbox
        return image[:, y1:y2, x1:x2]

    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        bboxA, bboxB = args

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.rand() > 0.5:
            a, b = b, a
            bboxA, bboxB = bboxB, bboxA

        # crop from b
        b = self.crop_bbox(b, bboxB)

        if self.shift > 0:
            _, h, w = a.shape
            pad = int(max(h, w) * self.shift)
            a_padding = torch.zeros(3, h+2*pad, w+2*pad)
            a_padding[:, pad:pad+h, pad:pad+w] = a
            offset_h = np.random.randint(0, 2*pad)
            offset_w = np.random.randint(0, 2*pad)
            a = a_padding[:, offset_h:offset_h+h, offset_w:offset_w+w]
            
            x1, y1, x2, y2 = bboxA
            x1 = max(0, x1 + pad - offset_w)
            y1 = max(0, y1 + pad - offset_h)
            x2 = min(w, x2 + pad - offset_w)
            y2 = min(h, y2 + pad - offset_h)
            bboxA = (x1, y1, x2, y2)
            
            if x1 == x2 or y1 == y2:
                return None
            
            # a[:, y1:y2, x1] = 1
            # a[:, y1:y2, x2] = 1
            # a[:, y1, x1:x2] = 1
            # a[:, y2, x1:x2] = 1
            
        if self.resize:
            scale = np.random.uniform(low=self.resize[0], high=self.resize[1])
            b = torch.nn.functional.interpolate(b.unsqueeze(0), scale_factor=scale, mode='bilinear').squeeze(0)
            
        # stamp b to a
        a_b, overlap = self.stamp(a, b, bboxA, self.max_overlap, self.max_iter)
        if overlap > self.max_overlap:
            return None

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b
            
class RatioMixer(Mixer):
    def __init__(self, channel_first=True, vertical=True, gap=0, jitter=3, shake=True):
        self.channel_first = channel_first
        self.vertical = vertical
        self.gap = gap
        self.jitter = jitter
        self.shake = shake

    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        gap = round(self.gap / 2)
        jitter = np.random.randint(-self.jitter, self.jitter + 1)

        if vertical:
            pivot = np.random.randint(0, w // 2 - jitter) if self.shake else w // 4 - jitter // 2
            a_b[:, :, :w // 2 + jitter - gap] = a[:, :, pivot:pivot + w // 2 + jitter - gap]
            pivot = np.random.randint(-jitter, w // 2) if self.shake else w // 4 - jitter // 2
            a_b[:, :, w // 2 + jitter + gap:] = b[:, :, pivot + jitter + gap:pivot + w // 2]
        else:
            pivot = np.random.randint(0, w // 2 - jitter) if self.shake else w // 4 - jitter // 2
            a_b[:, :, :w // 2 + jitter - gap] = a[:, :, pivot:pivot + w // 2 + jitter - gap]
            pivot = np.random.randint(-jitter, w // 2) if self.shake else w // 4 - jitter // 2
            a_b[:, :, w // 2 + jitter + gap:] = b[:, :, pivot + jitter + gap:pivot + w // 2]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b

class DiagnalMixer(Mixer):
    def __init__(self, channel_first=True, vertical=True):
        self.channel_first = channel_first
        self.vertical = vertical


    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        if vertical:
            for i in range(32):
                a_b[:,i,:w-i] = a [:,i,:w-i]
                a_b[:,i,w-i+1:] = b[:,i,w-i+1:]
        else:
            pivot = np.random.randint(0, h // 2 - jitter) if self.shake else h // 4 - jitter // 2
            a_b[:, :h // 2 + jitter - gap, :] = a[:, pivot:pivot + h // 2 + jitter - gap, :]
            pivot = np.random.randint(-jitter, h // 2) if self.shake else h // 4 - jitter // 2
            a_b[:, h // 2 + jitter + gap:, :] = b[:, pivot + jitter + gap:pivot + h // 2, :]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b


class DonutMixer(Mixer):
    def __init__(self, channel_first=True, vertical=True):
        self.channel_first = channel_first
        self.vertical = vertical


    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        if vertical:
            a_b = b
            a_b[:, h // 5 :4 * h // 5,  w // 5 :4 * w // 5 ] = a[:, h // 5 :4 * h // 5 ,  w // 5 :4 * w // 5 ]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b

class HotDogMixer(Mixer):
    def __init__(self, channel_first=True, vertical=True):
        self.channel_first = channel_first
        self.vertical = vertical


    def mix(self, a, b, *args):
        assert (self.channel_first and a.shape[0] <= 3) or (not self.channel_first and a.shape[-1] <= 3)
        assert a.shape == b.shape

        is_ndarray = isinstance(a, np.ndarray)

        if is_ndarray:
            dtype = a.dtype
            a = torch.FloatTensor(a)
            b = torch.FloatTensor(b)

        if not self.channel_first:
            a = a.permute(2, 0, 1)  # hwc->chw
            b = b.permute(2, 0, 1)

        if np.random.randint(0, 2):
            a, b = b, a

        a_b = torch.zeros_like(a)
        c, h, w = a.shape
        vertical = self.vertical or np.random.randint(0, 2)
        if vertical:
            a_b = a
            a_b[:, h // 4 :3 * h // 4 , :] = b[:, h // 4 :3 * h // 4, :]

        if not self.channel_first:
            a_b = a_b.permute(1, 2, 0)  # chw->hwc

        if is_ndarray:
            return a_b.data.numpy().copy().astype(dtype)
        else:
            return a_b