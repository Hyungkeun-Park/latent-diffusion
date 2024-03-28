import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import torchvision


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="shark",
        help="the class to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_trainable.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()

    #=======================================================

    test_model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    test_model.eval()
    #print(list(model.named_children()))

    #optimizer = torch.optim.SGD(model.cond_stage_model.transformer.ctx,lr=0.002,momentum=0.9,weight_decay=0.0005)
    trainable_prompt = model.cond_stage_model.transformer.ctx
    trainable = {'params':trainable_prompt}
    #trainable = {'params':model.cond_stage_model.transformer.ctx,}
    optimizer = torch.optim.SGD([trainable],lr=0.1,momentum=0.9,weight_decay=0.0005)

    #print(optimizer)

    #=======================================================

    iter=100

    for i in range(iter):
        #with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H//8, opt.W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                x_samples_ddim_copy = torch.clone(x_samples_ddim).detach()      #

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)
        #

        logit = test_model(x_samples_ddim_copy)
        
        ans = torch.nn.functional.softmax(logit,dim=1)
        print(f"\n\nTeacher's Class : {torch.topk(ans,1)}")

        #print(logit.size())
        # Class shark = 4
        target = torch.tensor([4]*opt.n_samples).to(device)

        optimizer.zero_grad()
        #print(f"\n\nPrompt before step :\n{model.cond_stage_model.transformer.ctx}\n\n")
        #print(f"\n\nPrompt before step :\n{model.cond_stage_model.transformer.ctx}\n\n")
        print(f"\n\nOptimizer's Zero Grad : {optimizer.param_groups[0]['params'][0].grad}")
        loss = torch.nn.functional.cross_entropy(logit, target)
        #print(f"\n\nLoss value : {loss}\n\n")
        loss.backward()
        
        #print(f"\n\nPrompt after step :\n{model.cond_stage_model.transformer.ctx}\n\n")
        #print(f"\n\nPrompt after step :\n{model.cond_stage_model.transformer.ctx}\n\n")
        print(f"\n\nOptimizer's Grad : {optimizer.param_groups[0]['params'][0].grad}")
        optimizer.step()




    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"\nYour samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
