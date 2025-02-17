import torch
import torch.nn as nn
import numpy as np
from models.siren import Siren, MLP
from utils import embed2affine
from scipy.ndimage import label
import scipy.ndimage as ndi

class CINA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sr_dims = sum(args['out_dim'][:-1])
        self.sr_net = Siren(args['in_dim'], args['latent_dim'][0] + args['cond_dims'], args['out_dim'], args['hidden_size'],
                            args['num_hidden_layers'], f_om=args['first_omega'], h_om=args['hidden_omega'],
                            outermost_linear=True, modulated_layers=args['modulated_layers'],
                            head_hidden_size=args['head_hidden_size'], head_num_layers=args['head_num_layers'],
                            head_omega=args['head_omega'])

    def forward(self, coords, latent_vecs, trafos=None):
        coords = self.transform(coords, trafos) if trafos is not None else coords
        input_tpl = (coords, latent_vecs)
        output = self.sr_net(input_tpl)
        return output

    def inference(self, coords, latent_vec, img_shape, trafos=None, step_size=50000):
        out_dim = sum(self.args['out_dim'])  # mri intensities and segmentation softmax
        output = torch.empty((coords.shape[0], out_dim)).to(self.args['device'])
        coords = self.transform(coords, trafos) if trafos is not None else coords
        for i in range(0, coords.shape[0], step_size):
            c = coords[i:i+step_size]
            lv = latent_vec[i:i+step_size]
            output[i:i + step_size] = self.forward(c, lv)
        imgs = output[..., :self.sr_dims].reshape(img_shape+[-1])
        seg_hard = torch.argmax(output[..., self.sr_dims:], dim=-1).reshape(img_shape+[-1])
        seg_soft = torch.nn.functional.softmax(output[..., self.sr_dims:], dim=-1).reshape(img_shape+[-1])
        modalities_rec = torch.cat((imgs.clip(0.0, 1.0), seg_hard, seg_soft), dim=-1)

        if self.args['mask_reconstruction']:
            return self.mask_reconstruction(modalities_rec, seg_hard)
        else:
            return modalities_rec

    @staticmethod
    def transform(coords, trafos, inverse=False):
        R, t = embed2affine(trafos)
        if inverse:
            R = R.inverse()
            t = -torch.einsum('nij,nj->ni', R, t)
        coords = torch.einsum('nxy,ny->nx', R, coords) + t
        return coords

    def mask_reconstruction(self, recs, seg):
        mask = self.connected_components(seg)
        mask = mask.expand_as(recs)
        return recs * mask

    def connected_components(self, seg):
        bg_label = self.args['label_names'].index('BG')
        mask = ((seg > 0) & (seg != bg_label)).detach().cpu().numpy()
        shp = np.array(list(mask.shape[:-1]))
        ps = ((shp * 0.1) //2).astype(int)  # patch size 10% of image size
        cp = shp // 2
        # get connected components
        labeled_mask, num_labels = label(mask)
        # get majority label of center patch
        center_label = labeled_mask[cp[0]-ps[0]:cp[0]+ps[0], cp[1]-ps[1]:cp[1]+ps[1], cp[2]-ps[2]:cp[2]+ps[2]].flatten()
        majority_label = np.argmax(np.bincount(center_label))
        # set all other labels to 0
        mask = mask * (labeled_mask == majority_label)

        # import matplotlib.pyplot as plt
        # print the middle slice of seg
        # plt.imshow(mask[cp[0], :, :, 0])
        # plt.show()
        mask_blur = (ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0) > 0.001).astype(np.uint8)
        # mask_dif = mask.astype(np.uint8) - mask_blur
        # plt.imshow(mask_dif[cp[0], :, :, 0])
        # plt.show()
        # seg_np = seg.detach().cpu().numpy()
        # seg_np[mask_dif.astype(bool)] = bg_label
        # plt.imshow(seg_np[cp[0], :, :, 0])
        # plt.show()
        # seg = torch.from_numpy(seg_np).to(self.args['device'], torch.float)
        return torch.from_numpy(mask_blur).to(self.args['device'], torch.float)
